"""Shared loader for the unified v2 cache (C3 foundation).

One entry point replaces the five near-identical ``load_cached_data`` copies.
Feature mode (which channels) is now a *load-time* argument, not a baked-in
property of the cache, so a single cache serves the 1-channel and 2-channel
tracks.

SCOPE NOTE: this C3 version reproduces the pipeline's CURRENT encoding — scalar
per-channel input standardization and linear per-species label z-scoring — so
switching to the frozen ID'd split changes *which data* a model sees, not *how
it is encoded*. The log/CLR label space (C2) and per-λ log-flux input norm (H3)
plug in here as alternative ``label_transform`` / ``input_norm`` options without
touching the cache or the tracks.

Statistics are ALWAYS estimated on the train split and reused for val/test,
never refit — the only leak-free choice.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from common.transforms import LabelPipeline

TARGET_COLUMNS = [
    "H2O", "CO2", "O2", "N2", "CH4", "N2O",
    "CO", "O3", "SO2", "NH3", "C2H6", "NO2",
]

# Channel indices within {split}_x.pt (0=star_planet_signal, 1=planet_signal).
_FEATURE_CHANNELS = {
    "planet": [1],
    "star_planet": [0],
    "both": [0, 1],
}

# Drop the 2.0 µm boundary bin (last λ index). Its noise≈0 makes the contrast
# there ~noise-free — a one-bin leak — and pins the stellar SNR at the clamp
# ceiling. Measured: bin 4378 is the only λ with noise==0 for >50% of planets.
# Applied at LOAD time so every consumer (train/eval/cloud) sees the same grid
# without rebuilding the 94 GB→cache.
TRIM_TAIL_BINS = 1

# Bump to invalidate on-disk DERIVED stats (norm, continuum, scalar/label stats)
# whenever the raw load changes shape/semantics (e.g. the tail trim above). Old
# files carry a different tag, so a stale full-length stat can never be silently
# reused; the missing tagged file simply regenerates.
DERIVED_CACHE_VERSION = "v2trim1"


def feature_channels(feature_mode):
    if feature_mode not in _FEATURE_CHANNELS:
        raise ValueError(f"feature_mode must be one of {list(_FEATURE_CHANNELS)}, got {feature_mode!r}")
    return _FEATURE_CHANNELS[feature_mode]


def _trim(t):
    """Drop the last TRIM_TAIL_BINS wavelength bins along the last axis."""
    return t[..., :-TRIM_TAIL_BINS] if TRIM_TAIL_BINS else t


def load_wavelength(cache_dir):
    """Shared λ grid (µm), tail-trimmed to match load_raw."""
    return _trim(torch.load(os.path.join(cache_dir, "wavelength.pt")))


def load_raw(cache_dir, split, feature_mode="planet"):
    """Return (x[N,C,L], y[N,12], noise[N,L], ids) — RAW, no transform.

    C1 (noise injection) and the causal track (env conditioning via ids/meta)
    build on this; the standardized path below wraps it. The tail boundary bin is
    trimmed (TRIM_TAIL_BINS) so L here is one shorter than the on-disk cache.
    """
    ch = feature_channels(feature_mode)
    x = torch.load(os.path.join(cache_dir, f"{split}_x.pt"))[:, ch, :].contiguous()
    y = torch.load(os.path.join(cache_dir, f"{split}_y.pt"))
    noise = torch.load(os.path.join(cache_dir, f"{split}_noise.pt"))
    with open(os.path.join(cache_dir, f"{split}_ids.json")) as f:
        ids = json.load(f)
    return _trim(x.float()).contiguous(), y.float(), _trim(noise.float()).contiguous(), ids


def load_meta(cache_dir, split):
    """summary rows for a split, in tensor row order (star_class, T_star, ...)."""
    return pd.read_csv(os.path.join(cache_dir, f"{split}_meta.csv"), dtype={"planet_index": str})


def _scalar_stats(cache_dir, feature_mode):
    """Scalar per-channel INPUT stats from TRAIN (the current encoding; H3 will
    add a per-λ log alternative). Cached to disk per feature_mode."""
    path = os.path.join(cache_dir, f"scalar_stats_{feature_mode}_{DERIVED_CACHE_VERSION}.pt")
    if os.path.exists(path):
        return torch.load(path)
    tr_x, _, _, _ = load_raw(cache_dir, "train", feature_mode)
    stats = {
        "mean_x": tr_x.mean(dim=(0, 2), keepdim=True),   # (1, C, 1)
        "std_x": tr_x.std(dim=(0, 2), keepdim=True),
    }
    torch.save(stats, path)
    return stats


def label_pipeline(cache_dir, label_transform="clr"):
    """Return the train-fitted LabelPipeline for `label_transform`, cached to disk.

    Labels are independent of feature_mode, so this is fit once per transform on
    the train split and reused for val/test and at eval time (for decode).
    """
    path = os.path.join(cache_dir, f"label_pipe_{label_transform}_{DERIVED_CACHE_VERSION}.pt")
    if os.path.exists(path):
        return LabelPipeline.from_state(torch.load(path))
    _, tr_y, _, _ = load_raw(cache_dir, "train", "planet")
    lp = LabelPipeline.fit(tr_y, label_transform)
    torch.save(lp.state(), path)
    return lp


def load_v2(cache_dir, split, feature_mode="planet", standardize=True, label_transform="clr"):
    """Return a TensorDataset(x, y_encoded) for one split.

    label_transform in {"linear","log10","clr"} (C2). "linear" reproduces the old
    linear-space z-scoring; "clr" (default) trains on the simplex-correct space.
    Decode model outputs at eval time with ``label_pipeline(cache_dir, kind).decode``.
    Parameters mirror the old ``load_cached_data`` so tracks migrate by swapping
    the import and passing a split name instead of relying on the internal 80/20.
    """
    x, y, _noise, _ids = load_raw(cache_dir, split, feature_mode)
    lp = label_pipeline(cache_dir, label_transform)
    y = lp.encode(y)
    if not standardize:
        return TensorDataset(x, y)
    s = _scalar_stats(cache_dir, feature_mode)
    x = (x - s["mean_x"]) / (s["std_x"] + 1e-30)
    return TensorDataset(x, y)
