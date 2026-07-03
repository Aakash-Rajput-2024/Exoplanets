"""Parametric grey-cloud transform for INARA reflected-light spectra.

PHYSICS
-------
In the 0.2-2.0 um window, INARA's `planet_signal` is REFLECTED starlight:

    F_p(lambda) ~ (R_p/a)^2 * A_g(lambda) * F_star(lambda) * Phase

The wavelength structure lives in the geometric albedo A_g, which is a smooth
CONTINUUM (Rayleigh scattering, surface/cloud reflectivity, stellar shape) with
downward DIPS where molecules (H2O, CO2, O2, CH4, ...) absorb the reflected
light. So a clear spectrum is "continuum minus absorption bands".

A (grey) cloud deck does two physical things in reflected light:
  1. MUTES the molecular bands. Photons reflect off the cloud top, traversing a
     shorter gas column, so absorption features shrink toward zero.
  2. BRIGHTENS the continuum. Clouds are bright, ~grey reflectors that raise the
     overall albedo.

MODEL (grey + uniform; v1)
--------------------------
Decompose the clear spectrum p(lambda):
    c(lambda) = continuum  (band-free upper envelope of p)
    d(lambda) = c - p >= 0 (band depth)

Two scalar parameters per spectrum:
    f in [0, 1]  cloud opacity / coverage  -> band-muting strength
    b >= 0       continuum brightening      -> cloud reflectivity boost

    p_mute  = c - (1 - f) * d = p + f * (c - p)      # f=0 -> p, f=1 -> c
    p_cloud = p_mute * (1 + b * f)                    # opaque clouds reflect more

This is a deliberately simple, physically-motivated AUGMENTATION (not a
radiative-transfer recomputation). It is GREY (f, b wavelength-independent) and
UNIFORM (all bands muted by the same fraction). Wavelength-dependent / molecule-
aware (cloud-top-pressure) refinements are deferred to v2.

ISOLATION: this module only reads spectra passed to it; it never touches the
original caches, checkpoints, or training code.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - scipy is expected in the train env
    _HAVE_SCIPY = False


# Default sampling ranges for augmentation (see module docstring).
F_RANGE = (0.3, 1.0)   # skip near-clear; the clear copy already covers f~0
B_RANGE = (0.0, 1.0)   # up to ~2x continuum brightening at full opacity

# Continuum-estimation hyperparameters (in grid points; 4379 pts over 0.2-2.0um).
CONTINUUM_MAX_WINDOW = 121   # wide enough to bridge molecular band widths
CONTINUUM_SMOOTH_SIGMA = 40.0


def estimate_continuum(
    p: np.ndarray,
    max_window: int = CONTINUUM_MAX_WINDOW,
    smooth_sigma: float = CONTINUUM_SMOOTH_SIGMA,
) -> np.ndarray:
    """Estimate the band-free continuum as a smoothed upper envelope of `p`.

    Molecular bands are downward dips, so the continuum rides the top: a maximum
    filter lifts the signal over band cores, then a Gaussian smooth de-jags it.

    Parameters
    ----------
    p : (..., L) array
        Raw spectra. The transform is applied along the last axis, so this works
        on a single spectrum (L,) or a batch (N, L).
    """
    p = np.asarray(p, dtype=np.float64)
    if not _HAVE_SCIPY:
        return _estimate_continuum_numpy(p, max_window, smooth_sigma)

    env = maximum_filter1d(p, size=max_window, axis=-1, mode="nearest")
    cont = gaussian_filter1d(env, sigma=smooth_sigma, axis=-1, mode="nearest")
    # The continuum must not dip below the signal (bands are absorption only).
    return np.maximum(cont, p)


def _estimate_continuum_numpy(p: np.ndarray, max_window: int, smooth_sigma: float) -> np.ndarray:
    """scipy-free fallback: sliding-window max + boxcar smoothing."""
    L = p.shape[-1]
    half = max_window // 2
    flat = p.reshape(-1, L)
    env = np.empty_like(flat)
    for i in range(L):
        lo = max(0, i - half)
        hi = min(L, i + half + 1)
        env[:, i] = flat[:, lo:hi].max(axis=1)
    # Gaussian-ish smoothing via repeated boxcar (cheap, adequate for fallback).
    k = max(1, int(smooth_sigma))
    kernel = np.ones(k) / k
    smoothed = np.empty_like(env)
    for r in range(env.shape[0]):
        padded = np.pad(env[r], k, mode="edge")
        smoothed[r] = np.convolve(padded, kernel, mode="same")[k:-k]
    cont = smoothed.reshape(p.shape)
    return np.maximum(cont, p)


def apply_cloud(
    p: np.ndarray,
    f: float,
    b: float,
    continuum: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the grey-cloud transform to a clear reflected-light spectrum.

    Parameters
    ----------
    p : (..., L) array
        Clear `planet_signal` spectrum/spectra.
    f : float in [0, 1]
        Cloud opacity (band-muting fraction). 0 = clear, 1 = bands removed.
    b : float >= 0
        Continuum brightening factor.
    continuum : array, optional
        Precomputed continuum (same shape as `p`). If omitted it is estimated.

    Returns
    -------
    p_cloud : array, same shape and dtype family as `p`.
    """
    p64 = np.asarray(p, dtype=np.float64)
    c = estimate_continuum(p64) if continuum is None else np.asarray(continuum, dtype=np.float64)
    p_mute = p64 + f * (c - p64)
    p_cloud = p_mute * (1.0 + b * f)
    return p_cloud.astype(p.dtype if hasattr(p, "dtype") else np.float32)


def sample_cloud_params(rng: np.random.Generator, f_range=F_RANGE, b_range=B_RANGE):
    """Draw (f, b) for one cloudy realization from a seeded RNG."""
    f = float(rng.uniform(*f_range))
    b = float(rng.uniform(*b_range))
    return f, b


def cloud_channel_for_mode(n_channels: int) -> int:
    """Which input channel carries reflected planet light (gets clouded).

    1ch caches: channel 0 is `planet_signal`.
    2ch caches: channel 0 is `star_planet_signal` (star-dominated, ~1e-9; clouds
    are invisible there), channel 1 is `planet_signal` (~1e-18; clouded).
    """
    return 0 if n_channels == 1 else 1


def _self_test():
    """Load one real INARA spectrum, apply the transform, and plot it.

    Read-only and instant: no training, no cache writes. Saves a PNG next to
    this file so the band-muting + brightening behaviour can be eyeballed.
    """
    import os
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = "/Users/aakashrajput/MachineLearning/Exoplanets/data/inara_1by3"
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    files.sort()
    sample = os.path.join(data_dir, files[0])
    df = pd.read_csv(sample)
    df = df[df["wavelength_(um)"] <= 2.0]
    wl = df["wavelength_(um)"].to_numpy()
    p = np.nan_to_num(pd.to_numeric(df["planet_signal_(erg/s/cm2)"], errors="coerce").to_numpy())

    c = estimate_continuum(p)
    print(f"Self-test spectrum: {files[0]}  ({len(p)} pts, <=2.0um)")
    print(f"  clear  : min={p.min():.3e} max={p.max():.3e}")
    print(f"  cont.  : min={c.min():.3e} max={c.max():.3e}  (>= clear everywhere: {bool((c >= p - 1e-30).all())})")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wl, p, lw=0.8, label="clear (f=0)", color="black")
    ax.plot(wl, c, lw=1.0, ls="--", label="estimated continuum", color="gray")
    for f, b, color in [(0.5, 0.3, "tab:blue"), (0.9, 0.6, "tab:red")]:
        pc = apply_cloud(p, f, b, continuum=c)
        # Quantify band muting: residual area below continuum.
        depth_frac = (c - pc).clip(min=0).sum() / max((c - p).clip(min=0).sum(), 1e-300)
        print(f"  f={f}, b={b}: residual band depth = {depth_frac:.2f} of clear")
        ax.plot(wl, pc, lw=0.8, label=f"cloud f={f}, b={b}", color=color)
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("planet_signal (erg/s/cm2)")
    ax.set_title("Grey-cloud transform: band muting + continuum brightening")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_model_selftest.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"=> Saved self-test plot to {out}")


if __name__ == "__main__":
    _self_test()
