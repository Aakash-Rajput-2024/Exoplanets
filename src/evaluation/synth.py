"""Assemble an in-memory INARA-scale (raw_x, noise) pair from engine spectrum
SHAPES — the in-env equivalent of crossgen_eval/build_cache_v2_prt.py.

Median-matches each channel to the real cache_v2 test medians (washes out arbitrary
albedo/radius/units while preserving spectral SHAPE — the informative part), sets
ch0 = star+planet, and bootstraps REAL INARA noise rows (so the leak-free stellar-SNR
channel stays in-distribution and the measured gap is isolated to the CONTRAST shape).

Grid: engine shapes are produced on the full INARA grid (load_inara_grid, 4379 pts);
the last boundary bin is trimmed here (common.data.TRIM_TAIL_BINS) so raw_x/noise match
the per-λ length the INARA-fit norm/label pipeline expect (4378).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_THIS, os.pardir))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from common.data import TRIM_TAIL_BINS   # noqa: E402

_MEDIAN_CACHE = {}
_NOISE_CACHE = {}


def _trim(a):
    return a[..., :-TRIM_TAIL_BINS] if TRIM_TAIL_BINS else a


def inara_channel_medians(cache_v2):
    """(median star_planet, median planet) from the real cache_v2 test tensor. Memoised."""
    if cache_v2 not in _MEDIAN_CACHE:
        x = torch.load(os.path.join(cache_v2, "test_x.pt"), map_location="cpu")   # [M,2,L]
        _MEDIAN_CACHE[cache_v2] = (float(x[:, 0, :].median()), float(x[:, 1, :].median()))
        del x
    return _MEDIAN_CACHE[cache_v2]


def inara_noise_pool(cache_v2, split="test"):
    """Real INARA per-bin 1σ noise rows [M, L] (trimmed). Memoised."""
    key = (cache_v2, split)
    if key not in _NOISE_CACHE:
        n = torch.load(os.path.join(cache_v2, f"{split}_noise.pt"), map_location="cpu").float()
        _NOISE_CACHE[key] = _trim(n).contiguous()
    return _NOISE_CACHE[key]


def _median_match(shape_batch, target_median):
    """Scale a [N,L] shape batch so its positive-value median == target_median."""
    arr = np.abs(np.asarray(shape_batch))
    pos = arr[arr > 0]
    k = float(target_median / np.median(pos)) if pos.size else 1.0
    return np.asarray(shape_batch) * k


def build_raw(planet_shapes, star_shapes, cache_v2, seed=0):
    """planet_shapes,[star_shapes]: [N, L_full] arrays on the full INARA grid.

    Returns (raw_x[N,2,L], noise[N,L]) as torch tensors on the trimmed grid, at the
    real INARA absolute scale — ready for core.predict_raw.
    """
    planet_shapes = np.atleast_2d(np.asarray(planet_shapes, dtype=float))
    star_shapes = np.atleast_2d(np.asarray(star_shapes, dtype=float))
    N = planet_shapes.shape[0]
    m_star, m_planet = inara_channel_medians(cache_v2)

    planet = _median_match(planet_shapes, m_planet)
    star = _median_match(star_shapes, m_star)
    star_planet = star + planet
    raw = np.stack([star_planet, planet], axis=1)            # [N,2,L_full]
    raw_x = _trim(torch.from_numpy(np.nan_to_num(raw)).float()).contiguous()

    pool = inara_noise_pool(cache_v2)                        # [M, L]
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool.shape[0], size=N, replace=(N > pool.shape[0]))
    noise = pool[idx].contiguous()
    assert raw_x.shape[2] == noise.shape[1], (raw_x.shape, noise.shape)
    return raw_x, noise
