"""Paired (clouded → clear) data for the declouder — READ-ONLY on the pipeline.

Because ``common.cloud_families`` is a known forward operator, every clear INARA
spectrum plus a cloud draw is an EXACT training pair. This module builds those
pairs in the retrieval model's own input space:

    input  = encoded NOISY clouded observable   [B, 2, L]  ([contrast, stellar-SNR])
    target = encoded NOISELESS clear contrast    [B, 1, L]  (the ideal restoration)

Both use the INARA-**train**-fit asinh norm (``common.pipeline.get_norm``), so the
declouder's output lands exactly where a frozen retrieval model expects its ch0.
The SNR channel is cloud-invariant (numerator F_star = star_planet − planet), so it
is passed through untouched as per-λ reliability context.

Isolation: reads ``data/cache_v2`` and ``common.*`` only; the sole thing written is
a reusable per-split continuum under ``src/decloud/cache/``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common.registry import CACHE_V2
from common.data import load_raw, load_wavelength, DERIVED_CACHE_VERSION
from common.pipeline import get_norm
from common.observable import make_observable, inject_noise
from common import cloud_families

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(THIS_DIR, "cache")

# Grey-cloud augmentation ranges for TRAINING (skip near-clear; the identity
# residual already covers f≈0). b brightens the continuum up to ~2×.
GREY_F_RANGE = (0.3, 1.0)
GREY_B_RANGE = (0.0, 1.0)
# Exposure augmentation over the eval-sweep range (matches registry.MATCHED), so
# the declouder is robust across noise levels, not tuned to one exposure.
DEFAULT_ALPHA_RANGE = (0.3, 300.0)

# grey = the family the declouder TRAINS on; the rest are held out to measure
# generalisation to unseen cloud physics (mirrors common.cloud_families' C5 idea).
FAMILIES = ("grey",) + cloud_families.HELD_OUT_FAMILIES


def _draw_log_uniform_alpha(n, alpha_range, generator):
    """Per-sample α ~ log-uniform[lo, hi], shape (n, 1). α = √(t/t_nom)."""
    lo, hi = alpha_range
    u = torch.rand(n, 1, generator=generator)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def get_continuum(cache_dir, split, chunk=2048):
    """Band-free continuum of a split's clear planet flux, cached under decloud/.

    Clouds don't change the continuum, so it's estimated once and reused across
    epochs/seeds. Tagged with DERIVED_CACHE_VERSION so a pipeline trim change can
    never silently reuse a stale continuum.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = os.path.basename(os.path.normpath(cache_dir))
    path = os.path.join(CACHE_DIR, f"continuum_{tag}_{split}_{DERIVED_CACHE_VERSION}.pt")
    if os.path.exists(path):
        return torch.load(path)
    x, _, _, _ = load_raw(cache_dir, split, feature_mode="both")
    cont = torch.from_numpy(cloud_families.estimate_continuum(x[:, 1, :].numpy(), chunk=chunk)).float()
    torch.save(cont, path)
    return cont


class CloudPairDataset(Dataset):
    """Clear (star_planet, planet, continuum, noise) rows for one cache split.

    Cheap views only; :class:`GreyCloudPairCollate` does the clouding/encoding per
    batch (the same fast pattern as ``common.inputs``).
    """

    def __init__(self, cache_dir=CACHE_V2, split="train", max_rows=None):
        x, _, noise, _ = load_raw(cache_dir, split, feature_mode="both")
        if max_rows is not None and x.shape[0] > max_rows:
            # Smoke/subset path: slice first, estimate the continuum on just the
            # subset (fast) instead of the cached full-split continuum.
            x, noise = x[:max_rows], noise[:max_rows]
            cont = torch.from_numpy(cloud_families.estimate_continuum(x[:, 1, :].numpy())).float()
        else:
            cont = get_continuum(cache_dir, split)
        self.sp = x[:, 0, :].contiguous()      # clear star_planet = F_star + F_p
        self.p = x[:, 1, :].contiguous()       # clear planet flux F_p
        self.noise = noise
        self.continuum = cont

    def __len__(self):
        return self.p.shape[0]

    def __getitem__(self, i):
        return i, self.sp[i], self.p[i], self.continuum[i], self.noise[i]


class GreyCloudPairCollate:
    """Per-batch (clouded → clear) builder for TRAINING.

    Draws grey (f, b) and exposure α per sample from a content-seeded RNG
    (reproducible + worker-safe), applies the grey cloud to F_p, rebuilds a
    consistent star_planet, injects noise at α, and encodes. Target is the
    noiseless clear contrast under the same norm.
    """

    def __init__(self, norm, obs_mode="contrast_snr", alpha=1.0,
                 alpha_range=DEFAULT_ALPHA_RANGE, base_seed=0,
                 f_range=GREY_F_RANGE, b_range=GREY_B_RANGE):
        self.norm = norm
        self.obs_mode = obs_mode
        self.alpha = alpha
        self.alpha_range = alpha_range
        self.base_seed = base_seed
        self.f_range = f_range
        self.b_range = b_range
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _gen(self, idxs):
        key = self.base_seed * 1_000_003 + self.epoch * 9973 + int(sum(idxs))
        return torch.Generator().manual_seed(key % (2 ** 31 - 1))

    def _alpha(self, n, gen):
        if self.alpha_range is None:
            return self.alpha
        return _draw_log_uniform_alpha(n, self.alpha_range, gen)

    def __call__(self, batch):
        idxs = [b[0] for b in batch]
        sp = torch.stack([b[1] for b in batch])       # (B, L) clear star_planet
        p = torch.stack([b[2] for b in batch])        # (B, L) clear F_p
        cont = torch.stack([b[3] for b in batch])     # (B, L) continuum
        ns = torch.stack([b[4] for b in batch])       # (B, L) noise
        gen = self._gen(idxs)
        B = p.shape[0]
        f = self.f_range[0] + (self.f_range[1] - self.f_range[0]) * torch.rand(B, 1, generator=gen)
        b = self.b_range[0] + (self.b_range[1] - self.b_range[0]) * torch.rand(B, 1, generator=gen)
        f_star = sp - p
        p_cloud = (p + f * (cont - p)) * (1.0 + b * f)   # grey: mute bands + brighten
        sp_cloud = f_star + p_cloud
        a = self._alpha(B, gen)
        x_cloud = torch.stack([sp_cloud, p_cloud], dim=1)          # (B, 2, L)
        inp = self.norm.encode(inject_noise(x_cloud, ns, self.obs_mode, alpha=a, generator=gen))
        x_clear = torch.stack([sp, p], dim=1)
        tgt = self.norm.encode(make_observable(x_clear, ns, self.obs_mode))[:, 0:1, :]
        return inp, tgt


def apply_family_to_raw(raw_x, family, wl, continuum=None):
    """Return a cloudy copy of ``raw_x`` (F_p clouded, star_planet rebuilt).

    ``raw_x`` : (N, 2, L) [star_planet, planet].  ``wl`` : (L,) µm grid.
    Deterministic (representative family params from ``cloud_families.FAMILY_PARAMS``)
    — used by the downstream eval so every model sees the same clouded test set.
    """
    sp, p = raw_x[:, 0, :], raw_x[:, 1, :]
    c = cloud_families.estimate_continuum(p.numpy()) if continuum is None else np.asarray(continuum)
    p_cloud = torch.from_numpy(cloud_families.apply_family(family, p.numpy(), c, np.asarray(wl))).float()
    f_star = sp - p
    return torch.stack([f_star + p_cloud, p_cloud], dim=1)


def build_fixed_pairs(cache_dir, split, norm, family="grey", alpha=30.0,
                      noise_seed=101, n_max=None, obs_mode="contrast_snr"):
    """Deterministic (input, target) tensors for a split under one cloud family.

    Used for a STABLE validation metric during training and for reconstruction
    scoring in eval. ``input`` = encoded noisy clouded observable at fixed α/seed;
    ``target`` = encoded noiseless clear contrast.
    """
    x, _, noise, _ = load_raw(cache_dir, split, feature_mode="both")
    if n_max is not None and x.shape[0] > n_max:
        x, noise = x[:n_max], noise[:n_max]
    wl = load_wavelength(cache_dir).numpy()
    x_cloud = apply_family_to_raw(x, family, wl)
    gen = torch.Generator().manual_seed(int(noise_seed))
    inp = norm.encode(inject_noise(x_cloud, noise, obs_mode, alpha=alpha, generator=gen))
    tgt = norm.encode(make_observable(x, noise, obs_mode))[:, 0:1, :]
    return inp, tgt


def get_norm_v2(cache_dir=CACHE_V2, obs_mode="contrast_snr", input_norm="per_lambda_asinh"):
    """Thin pass-through so callers import one name from this module."""
    return get_norm(cache_dir, obs_mode, input_norm)
