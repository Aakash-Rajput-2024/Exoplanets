"""T2 forward model — a neural emulator of PSG trained on INARA.

WHY AN EMULATOR AT ALL
    T0 is exact but dies above α≈3 (importance weights collapse to 1-NN). To bound the
    ceiling in the high-α / noiseless regime — where the headline ~0.55 R² actually
    lives — a sampler must be able to evaluate the forward model at ARBITRARY θ, not
    only at the 88k library points. The repo's two physical engines are both wrong for
    that job: ``prt_engine`` is thermal emission over 5 species with 0.2–0.3 µm flat
    edge-filled and is flagged in AUDIT_REPORT.md §2.4 as ~360× off PSG in the H₂O
    bands, and ``reflected_engine`` is an explicitly-labelled band-template proxy. A
    retrieval built on either measures that engine, not INARA.

    An emulator trained on INARA is PSG-faithful by construction, and — the part that
    matters — its adequacy is directly falsifiable. See ``chi2_adequacy``.

THE ADEQUACY TEST IS THE WHOLE ARGUMENT
    An emulator is likelihood-exact at exposure α iff its residual is small compared to
    the observational noise at that α:

        χ²_emul(α) = mean_λ [ (C_emul − C_PSG) / (σ_C/α) ]²   ≪ 1

    χ² ≪ 1 means the emulator error is invisible under the noise and a retrieval using
    it IS a PSG retrieval. As α grows the noise shrinks and χ² rises, so there is some
    α above which the emulator is no longer adequate — and that boundary is T2's
    validity range, exactly as ESS is T0's. Measure it, publish it, and refuse to quote
    T2 numbers beyond it. An emulator whose χ² is never ≪1 is a failed tier, not a
    result to be explained away.

DESIGN
    The output is 4378-dim, far too wide to regress directly, and highly redundant. We
    take an SVD basis of the asinh-scaled training contrasts (asinh, not log, because
    contrast can be driven negative — the same reason common/inputs.py uses it) and
    regress only the K leading coefficients from θ with an MLP. K is a tunable
    fidelity/speed knob; ``basis_truncation_chi2`` reports the floor that the basis
    ALONE imposes, so a bad χ² can be attributed to the basis or to the MLP.

    θ is the 12 ILR-free VMRs in CLR space plus the physical nuisances that actually
    move a reflected-light spectrum (surface T/P, radius, density, albedo, star T and
    radius, semi-major axis). Those come from {split}_meta.csv, which is stored in
    tensor row order.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common.data import load_raw, load_meta, TARGET_COLUMNS
from common.observable import contrast, contrast_sigma

# Physical nuisances that move a reflected-light spectrum. Deliberately excludes
# distance_parsec (contrast is distance-independent) and the T-P shape parameters
# kappa/gamma1/gamma2/alpha/beta (they set the profile PSG integrates, but are not
# retrievable from a reflected-light spectrum and would only add unconstrained dims).
NUISANCE = ["surface_temperature", "surface_pressure", "planet_radius", "planet_density",
            "albedo", "star_temperature", "star_radius", "semimajor_axis"]
LOG_NUISANCE = {"surface_pressure", "planet_density", "semimajor_axis"}
LOG_FLOOR = 1e-12


def build_theta(meta: pd.DataFrame, y_lin) -> np.ndarray:
    """[N, 12 + len(NUISANCE)] design matrix: CLR composition + (log-)scaled nuisances.

    CLR rather than ILR here on purpose: the emulator is a plain regressor and never
    puts a density on θ, so the rank-11 redundancy is harmless, and CLR keeps each
    coordinate interpretable as one species. The SAMPLER (nested.py) uses ILR.
    """
    y = np.clip(np.asarray(y_lin, dtype=np.float64), LOG_FLOOR, None)
    ly = np.log(y)
    clr = ly - ly.mean(axis=1, keepdims=True)
    cols = []
    for c in NUISANCE:
        v = meta[c].to_numpy(dtype=np.float64)
        cols.append(np.log10(np.clip(v, 1e-30, None)) if c in LOG_NUISANCE else v)
    return np.concatenate([clr, np.stack(cols, axis=1)], axis=1)


class SpectrumEmulator(nn.Module):
    """θ → contrast(λ), via an MLP onto K SVD coefficients of asinh(C/s)."""

    def __init__(self, n_theta, n_lambda, K=96, hidden=(512, 512, 512)):
        super().__init__()
        layers, d = [], n_theta
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers += [nn.Linear(d, K)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("theta_mean", torch.zeros(1, n_theta))
        self.register_buffer("theta_std", torch.ones(1, n_theta))
        self.register_buffer("basis", torch.zeros(K, n_lambda))     # V^T
        self.register_buffer("coef_mean", torch.zeros(1, K))
        self.register_buffer("coef_std", torch.ones(1, K))
        self.register_buffer("spec_mean", torch.zeros(1, n_lambda))
        self.register_buffer("scale", torch.ones(1, n_lambda))      # asinh softening

    # --- the asinh encoding the basis lives in -----------------------------------
    def _encode_spec(self, C):
        return torch.asinh(C / self.scale) - self.spec_mean

    def _decode_spec(self, t):
        return torch.sinh(t + self.spec_mean) * self.scale

    def forward(self, theta):
        u = (theta - self.theta_mean) / (self.theta_std + 1e-8)
        coef = self.net(u) * (self.coef_std + 1e-8) + self.coef_mean
        return self._decode_spec(coef @ self.basis)

    @torch.no_grad()
    def project(self, C):
        """Best achievable reconstruction of C in this basis — the truncation floor."""
        t = self._encode_spec(C)
        return self._decode_spec((t @ self.basis.T) @ self.basis)


def fit_basis(C_train, K=96, n_svd=8000, seed=0):
    """SVD basis of the asinh-encoded training contrasts.

    Returns (basis [K, L], spec_mean [1, L], scale [1, L]). ``scale`` is the per-λ
    median |C| used to soften the asinh, mirroring ``inputs.PerLambdaAsinhNorm`` so the
    emulator and the network see the same conditioning of the spectral dynamic range.
    """
    scale = C_train.abs().median(dim=0, keepdim=True).values.clamp(min=1e-20)
    t = torch.asinh(C_train / scale)
    spec_mean = t.mean(dim=0, keepdim=True)
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(t.shape[0], generator=g)[:min(n_svd, t.shape[0])]
    _, _, Vh = torch.linalg.svd((t[sel] - spec_mean).double(), full_matrices=False)
    return Vh[:K].float().contiguous(), spec_mean, scale


def chi2_adequacy(model, theta, C_true, sigma_alpha1, alphas=(1, 3, 10, 30, 100, 300),
                  batch=512, use_projection=False):
    """χ²_emul per α: mean_λ [(C_emul − C_PSG)/(σ_C/α)]².

    ``use_projection=True`` swaps the MLP for the exact SVD projection, isolating the
    basis-truncation floor from the regression error — if the projection alone already
    fails, widening the MLP cannot help and K must go up.

    Returns {alpha: {"chi2_median", "chi2_mean", "adequate"}}; adequate ⇔ median < 0.1.
    """
    with torch.no_grad():
        preds = []
        for i in range(0, theta.shape[0], batch):
            preds.append(model.project(C_true[i:i + batch]) if use_projection
                         else model(theta[i:i + batch]))
        pred = torch.cat(preds)
    resid = (pred - C_true)
    out = {}
    for a in alphas:
        r = (resid / torch.clamp(sigma_alpha1 / float(a), min=1e-30)) ** 2
        med = float(r.mean(dim=1).median())
        out[float(a)] = {"chi2_median": med, "chi2_mean": float(r.mean()),
                         "adequate": bool(med < 0.1)}
    return out


def load_training_arrays(cache_v2, split="train", max_n=None):
    """(theta [N, P], C [N, L], sigma_a1 [N, L]) ready for fit/eval."""
    x, y, noise, _ = load_raw(cache_v2, split, feature_mode="both")
    meta = load_meta(cache_v2, split)
    if max_n is not None and x.shape[0] > max_n:
        x, y, noise, meta = x[:max_n], y[:max_n], noise[:max_n], meta.iloc[:max_n]
    C = contrast(x[:, 0, :], x[:, 1, :])
    sig = contrast_sigma(x[:, 0, :], x[:, 1, :], noise)
    theta = torch.as_tensor(build_theta(meta, y.numpy()), dtype=torch.float32)
    return theta, C, sig


def train_emulator(cache_v2, K=96, epochs=30, batch_size=256, lr=1e-3, seed=0,
                   max_n=None, device=None, hidden=(512, 512, 512), out=None):
    """Fit the emulator; returns (model, history). Validation is χ²-based, not MSE."""
    torch.manual_seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    th_tr, C_tr, _ = load_training_arrays(cache_v2, "train", max_n)
    th_va, C_va, sg_va = load_training_arrays(cache_v2, "val", None)

    basis, spec_mean, scale = fit_basis(C_tr, K=K, seed=seed)
    model = SpectrumEmulator(th_tr.shape[1], C_tr.shape[1], K=K, hidden=hidden)
    model.basis.copy_(basis); model.spec_mean.copy_(spec_mean); model.scale.copy_(scale)
    model.theta_mean.copy_(th_tr.mean(0, keepdim=True))
    model.theta_std.copy_(th_tr.std(0, keepdim=True))

    with torch.no_grad():
        coef_tr = (torch.asinh(C_tr / scale) - spec_mean) @ basis.T
        model.coef_mean.copy_(coef_tr.mean(0, keepdim=True))
        model.coef_std.copy_(coef_tr.std(0, keepdim=True))
        tgt = (coef_tr - model.coef_mean) / (model.coef_std + 1e-8)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = torch.utils.data.TensorDataset(th_tr, tgt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    hist = []
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for tb, yb in dl:
            tb, yb = tb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            u = (tb - model.theta_mean) / (model.theta_std + 1e-8)
            loss = nn.functional.mse_loss(model.net(u), yb)
            loss.backward()
            opt.step()
            tot += float(loss); n += 1
        sched.step()
        model.eval()
        chi = chi2_adequacy(model.cpu(), th_va[:2000], C_va[:2000], sg_va[:2000],
                            alphas=(1, 10, 100))
        model.to(device)
        hist.append({"epoch": ep, "coef_mse": tot / max(n, 1),
                     "chi2": {str(k): v["chi2_median"] for k, v in chi.items()}})
        print(f"[emul] ep {ep:>3} coef-mse {tot/max(n,1):>9.5f}  "
              + "  ".join(f"χ²(α={k:g})={v['chi2_median']:.3g}" for k, v in chi.items()),
              flush=True)

    model = model.cpu().eval()
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_theta": int(th_tr.shape[1]), "n_lambda": int(C_tr.shape[1]),
                               "K": K, "hidden": list(hidden), "nuisance": NUISANCE},
                    "history": hist}, out)
    return model, hist


def load_emulator(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    m = SpectrumEmulator(c["n_theta"], c["n_lambda"], K=c["K"], hidden=tuple(c["hidden"]))
    m.load_state_dict(blob["state_dict"])
    return m.eval(), c
