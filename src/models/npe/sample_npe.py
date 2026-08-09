"""Sample the trained NPE flow → posteriors in the shape the eval battery consumes.

Emits ``samples_log [S, N, 12]`` (log10 VMR), which is exactly what
``evaluation.metrics_extra`` (PIT / SBC / TARP / coverage / reliability) expects, and a
``point_log [N, 12]`` posterior mean IN LOG10 SPACE — the metric-matched point estimate,
since the pipeline scores R² of log10(VMR) and the Bayes-optimal estimator for that is
E[log10 θ | x], not log10 E[θ | x].

SBC WARNING, and it is easy to get wrong
    ``samples_for_sbc`` deliberately draws a FRESH noise realization per planet rather
    than reusing TEST_NOISE_SEED. Simulation-based calibration asks whether the truth is
    uniformly distributed within the posterior across draws from the joint p(θ, x); if
    every planet is conditioned on the same frozen noise seed, the ranks are conditioned
    on one realization and the resulting "calibration" is not SBC. Use
    ``samples_at_seed`` (fixed seed) for anything that must line up planet-for-planet
    with Section A, and ``samples_for_sbc`` for the calibration battery.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from common.data import load_raw                            # noqa: E402
from common.observable import inject_noise                  # noqa: E402
from common.pipeline import get_norm, TEST_NOISE_SEED       # noqa: E402
from common.registry import CACHE_V2                        # noqa: E402
from common.runtime import get_device                       # noqa: E402
from common.transforms import from_ilr, helmert_basis       # noqa: E402
from models.npe.train_npe import NPE, CKPT_DIR              # noqa: E402

LOG_FLOOR = 1e-12


def load_npe(ckpt_path=None, seed=0, device=None):
    device = device or get_device()
    path = ckpt_path or os.path.join(CKPT_DIR, f"npe_seed{seed}.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    m = NPE(in_channels=c["in_channels"], embed_dim=c["embed_dim"],
            transforms=c["transforms"])
    m.load_state_dict(blob["state_dict"])
    return m.to(device).eval(), c


def _encode(raw_x, noise, alpha, cfg, cache_v2, seed):
    norm = get_norm(cache_v2, cfg["obs_mode"], cfg["input_norm"])
    gen = torch.Generator().manual_seed(int(seed))
    return norm.encode(inject_noise(raw_x, noise, cfg["obs_mode"], alpha=alpha, generator=gen))


@torch.no_grad()
def _sample_encoded(model, x, alpha, n_samples, device, batch=64):
    """[S, N, 12] linear VMR from an already-encoded observable."""
    H = helmert_basis(12, dtype=torch.float32, device=device)
    out = []
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(device)
        la = torch.full((xb.shape[0],), float(np.log10(alpha)), device=device)
        z = model.sample(xb, la, n_samples)                  # [S, b, 11]
        out.append(from_ilr(z, H).cpu())                     # [S, b, 12] on the simplex
    return torch.cat(out, dim=1).numpy()


def samples_at_seed(model, cfg, cache_v2=CACHE_V2, split="test", alpha=1.0,
                    n_samples=200, seed=TEST_NOISE_SEED, idx=None, device=None):
    """Posterior samples on a FIXED noise realization — comparable to Section A.

    Noise is injected on the full split tensor before subsampling, matching
    ``build_eval_observable``; ``idx`` then selects the same planets Section K uses.
    """
    device = device or get_device()
    raw_x, y, noise, _ = load_raw(cache_v2, split, feature_mode="both")
    x = _encode(raw_x, noise, alpha, cfg, cache_v2, seed)
    if idx is not None:
        x, y = x[torch.as_tensor(idx)], y[torch.as_tensor(idx)]
    s = _sample_encoded(model, x, alpha, n_samples, device)
    return _pack(s, y.numpy())


def samples_for_sbc(model, cfg, cache_v2=CACHE_V2, split="test", alpha=1.0,
                    n_samples=200, n_planets=1000, seed=0, device=None):
    """Posterior samples with an INDEPENDENT noise draw per planet — valid for SBC.

    See the module docstring: reusing one fixed seed here would condition the rank
    statistics on a single noise realization and silently invalidate the diagnostic.
    """
    device = device or get_device()
    raw_x, y, noise, _ = load_raw(cache_v2, split, feature_mode="both")
    rng = np.random.default_rng(seed)
    sel = rng.choice(raw_x.shape[0], size=min(n_planets, raw_x.shape[0]), replace=False)
    sel.sort()
    t = torch.as_tensor(sel)
    raw_x, y, noise = raw_x[t], y[t], noise[t]
    gen = torch.Generator().manual_seed(int(seed) + 7919)
    norm = get_norm(cache_v2, cfg["obs_mode"], cfg["input_norm"])
    x = norm.encode(inject_noise(raw_x, noise, cfg["obs_mode"], alpha=alpha, generator=gen))
    s = _sample_encoded(model, x, alpha, n_samples, device)
    out = _pack(s, y.numpy())
    out["idx"] = sel
    return out


def _pack(samples_lin, y_lin):
    s_log = np.log10(np.clip(samples_lin, LOG_FLOOR, None))
    return {
        "samples_log": s_log,                       # [S, N, 12]
        "point_log": s_log.mean(axis=0),            # E[log10 θ | x] — metric-matched
        "truth_log": np.log10(np.clip(y_lin, LOG_FLOOR, None)),
    }
