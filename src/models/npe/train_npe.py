"""T1 — amortized neural posterior estimation on INARA (Vasist+2023, A&A 672 A147).

WHAT THIS IS
    A conditional normalizing flow q(θ | x, α) trained by maximum likelihood on the
    INARA training pairs. Once trained it emits a full posterior for any spectrum in
    milliseconds, for every α, on all 11,034 test planets — which is the speed/accuracy
    argument for ML retrieval, and the only tier that produces posteriors in the
    high-α regime where T0's importance sampling has collapsed to 1-NN.

    It is NOT an information ceiling. NPE is itself a neural estimator: if it plateaus,
    that is equally consistent with "the data is exhausted" and "neural estimators share
    a blind spot". Its posteriors are only trustworthy to the extent they pass SBC/TARP
    (``evaluation.metrics_extra``), so nothing from here should be quoted before those
    diagnostics are run — an under-trained flow looks fine in R² and is badly wrong in
    width.

PARAMETER SPACE — 11-dim ILR, not 12-dim CLR
    CLR is exactly zero-sum, so a density over its 12 coordinates is singular and the
    flow will diverge. ``common.transforms.to_ilr`` rotates to an orthonormal basis of
    the zero-sum hyperplane: 11 unconstrained coordinates, non-degenerate density, and
    ``from_ilr`` maps ANY finite point back to a strictly-positive composition summing
    to 1. Every posterior sample is therefore a valid simplex point by construction —
    the same guarantee CLR gives the regressors.

WHY NOT sbi's SNPE DRIVER
    ``sbi`` is installed and its flow (zuko) is what is used here, but its
    ``append_simulations`` API wants every (θ, x) pair materialized up front. One noise
    draw per training planet is 88,271 × 2 × 4378 × 4 B ≈ 3.1 GB, and fixing a single
    noise realization per planet would let the flow memorize the noise. Training
    directly against ``common.inputs``' streaming collate gives FRESH noise and a fresh
    log-uniform α every epoch, which is exactly how the regressors are trained — so the
    comparison stays honest and the memory stays flat.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 3))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

import zuko                                                            # noqa: E402
from common.data import load_raw                                       # noqa: E402
from common.inputs import RawObservableDataset, _draw_log_uniform_alpha  # noqa: E402
from common.observable import inject_noise                             # noqa: E402
from common.pipeline import get_norm, VAL_NOISE_SEED                   # noqa: E402
from common.registry import MATCHED, CACHE_V2                          # noqa: E402
from common.runtime import get_device                                  # noqa: E402
from common.transforms import to_ilr, helmert_basis                    # noqa: E402
from models.npe.embedding import SpectrumEmbedding                     # noqa: E402

CKPT_DIR = os.path.join(REPO, "src", "models", "npe", "checkpoints")


class AlphaNoiseCollate:
    """Like ``common.inputs.NoiseCollate`` but also returns the drawn α.

    The seed derivation is copied verbatim from NoiseCollate (content-derived, so it is
    reproducible and multiprocessing-safe). The flow needs α as an explicit conditioning
    variable, which the base collate does not surface.
    """

    def __init__(self, obs_mode, norm, alpha_range, base_seed=0):
        self.obs_mode, self.norm = obs_mode, norm
        self.alpha_range, self.base_seed, self.epoch = alpha_range, base_seed, 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __call__(self, batch):
        idxs = [b[0] for b in batch]
        xs = torch.stack([b[1] for b in batch])
        ns = torch.stack([b[2] for b in batch])
        ys = torch.stack([b[3] for b in batch])
        key = self.base_seed * 1_000_003 + self.epoch * 9973 + int(sum(idxs))
        gen = torch.Generator().manual_seed(key % (2 ** 31 - 1))
        a = _draw_log_uniform_alpha(xs.shape[0], self.alpha_range, gen)
        obs = self.norm.encode(inject_noise(xs, ns, self.obs_mode, alpha=a, generator=gen))
        return obs, torch.log10(a).squeeze(-1), ys


class NPE(nn.Module):
    """Embedding + conditional NSF over 11 standardized ILR coordinates."""

    def __init__(self, in_channels=2, embed_dim=256, transforms=8, hidden=(256, 256),
                 bins=8, ilr_mean=None, ilr_std=None):
        super().__init__()
        self.embed = SpectrumEmbedding(in_channels, embed_dim)
        self.flow = zuko.flows.NSF(features=11, context=self.embed.context_dim,
                                   transforms=transforms, hidden_features=hidden, bins=bins)
        self.register_buffer("ilr_mean", torch.zeros(1, 11) if ilr_mean is None else ilr_mean)
        self.register_buffer("ilr_std", torch.ones(1, 11) if ilr_std is None else ilr_std)

    def standardize(self, z):
        return (z - self.ilr_mean) / (self.ilr_std + 1e-8)

    def unstandardize(self, u):
        return u * (self.ilr_std + 1e-8) + self.ilr_mean

    def loss(self, x, log_alpha, z_ilr):
        return -self.flow(self.embed(x, log_alpha)).log_prob(self.standardize(z_ilr)).mean()

    @torch.no_grad()
    def sample(self, x, log_alpha, n):
        u = self.flow(self.embed(x, log_alpha)).sample((n,))     # [n, B, 11]
        return self.unstandardize(u)


def train(cache_v2=CACHE_V2, out=CKPT_DIR, epochs=40, batch_size=128, lr=1e-3,
          seed=0, patience=8, n_train=None, device=None, embed_dim=256,
          transforms=8, val_batches=40):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or get_device()
    obs_mode, in_norm = MATCHED["obs_mode"], MATCHED["input_norm"]
    arange = tuple(MATCHED["alpha_train_range"])

    tr_x, tr_y, tr_n, _ = load_raw(cache_v2, "train", feature_mode="both")
    va_x, va_y, va_n, _ = load_raw(cache_v2, "val", feature_mode="both")
    if n_train:
        tr_x, tr_y, tr_n = tr_x[:n_train], tr_y[:n_train], tr_n[:n_train]
    norm = get_norm(cache_v2, obs_mode, in_norm)

    H = helmert_basis(12, dtype=torch.float32)
    z_tr, z_va = to_ilr(tr_y, H), to_ilr(va_y, H)
    model = NPE(in_channels=tr_x.shape[1], embed_dim=embed_dim, transforms=transforms,
                ilr_mean=z_tr.mean(0, keepdim=True), ilr_std=z_tr.std(0, keepdim=True)
                ).to(device)

    coll = AlphaNoiseCollate(obs_mode, norm, arange, base_seed=seed)
    dl = DataLoader(RawObservableDataset(tr_x, tr_n, z_tr), batch_size=batch_size,
                    shuffle=True, collate_fn=coll, drop_last=True)
    # Val uses a FIXED seed and a fixed per-sample α draw so model selection is not
    # chasing noise, mirroring how the regressors' val stream is built.
    vcoll = AlphaNoiseCollate(obs_mode, norm, arange, base_seed=VAL_NOISE_SEED)
    vdl = DataLoader(RawObservableDataset(va_x, va_n, z_va), batch_size=batch_size,
                     shuffle=False, collate_fn=vcoll)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, f"npe_seed{seed}.pt")
    best, bad, hist = math.inf, 0, []

    for ep in range(epochs):
        model.train()
        coll.set_epoch(ep)
        t0, tot, nb = time.time(), 0.0, 0
        for x, la, z in dl:
            x, la, z = x.to(device), la.to(device), z.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(x, la, z)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MATCHED["grad_clip"])
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        model.eval()
        vcoll.set_epoch(0)
        with torch.no_grad():
            # islice, not `enumerate(...) if i < val_batches`: the latter still pulls
            # every val batch through the collate (noise injection + per-λ asinh encode
            # on 11k spectra) and throws most away. islice stops the loader instead.
            vs = [float(model.loss(x.to(device), la.to(device), z.to(device)))
                  for x, la, z in islice(vdl, val_batches)]
        v = float(np.mean(vs))
        hist.append({"epoch": ep, "train_nll": tot / max(nb, 1), "val_nll": v,
                     "sec": round(time.time() - t0, 1)})
        print(f"[npe seed{seed}] ep {ep:>3} train {tot/max(nb,1):>9.4f} "
              f"val {v:>9.4f} {hist[-1]['sec']:>6.1f}s"
              + ("  *best" if v < best else ""), flush=True)

        if v < best - 1e-4:
            best, bad = v, 0
            torch.save({"state_dict": model.state_dict(), "val_nll": v, "epoch": ep,
                        "config": {"in_channels": int(tr_x.shape[1]), "embed_dim": embed_dim,
                                   "transforms": transforms, "obs_mode": obs_mode,
                                   "input_norm": in_norm, "alpha_train_range": list(arange),
                                   "seed": seed}}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {ep} (best val {best:.4f})")
                break

    with open(os.path.join(out, f"npe_seed{seed}_log.json"), "w") as f:
        json.dump({"history": hist, "best_val_nll": best, "ckpt": ckpt}, f, indent=1)
    return ckpt, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-v2", default=CACHE_V2)
    ap.add_argument("--out", default=CKPT_DIR)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--transforms", type=int, default=8)
    a = ap.parse_args(argv)
    ckpt, best = train(a.cache_v2, a.out, epochs=a.epochs, batch_size=a.batch_size,
                       lr=a.lr, seed=a.seed, n_train=a.n_train, transforms=a.transforms)
    print(f"\nbest val NLL {best:.4f} -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
