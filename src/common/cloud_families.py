"""Held-out cloud families for the transfer test (fixes C5).

C5's defect: the flagship "cloud robustness" number trained AND evaluated on the
same grey+uniform transform, so the model merely learned to invert the exact
augmentation it saw — circular. The fix is to keep the grey family for *training*
but measure accuracy on cloud physics the model has **never seen**. The gap
between seen (grey) and unseen families is itself the publishable quantity.

All transforms act on the reflected-light planet flux F_p(λ) (the numerator of
the contrast observable; the star flux and thus the SNR channel are unchanged by
clouds). A clear spectrum is "continuum c(λ) minus absorption band depth
d(λ)=c−p". The grey-training model has only ever seen wavelength-independent,
spatially-uniform muting; each family below breaks one of those assumptions:

  grey            p + f·d,  brightened (1+b·f)          — the TRAINING family (ref)
  non_grey        f(λ)=f₀(λ/λ₀)^−α : blue-muted haze     — breaks λ-independence
  band_selective  strong saturated cores mute LESS than wings (cloud-top P proxy)
                  — breaks uniform-muting; the shortcut "all bands scale together"
  patchy          φ·cloudy + (1−φ)·clear, φ~Beta         — breaks φ=1; Earth is φ≈0.6

Self-contained (only numpy/scipy) so `common/` never depends on
`src/cloud_robust/`; the continuum estimator mirrors that module's but is chunked
over planets to bound memory on the 11k×4379 test tensor.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

CONTINUUM_MAX_WINDOW = 121
CONTINUUM_SMOOTH_SIGMA = 40.0

# Representative parameters per family (documented so the transfer table is
# reproducible). Chosen at moderate-to-heavy opacity where families differ most.
FAMILY_PARAMS = {
    "grey":           dict(f=0.7, b=0.5),
    "non_grey":       dict(f0=0.7, alpha=2.0, b0=0.5),
    "band_selective": dict(f=0.7, gamma=1.5, b=0.5),
    "patchy":         dict(phi=0.6, f=0.9, b=0.5),   # Earth-like fractional cover
}


def estimate_continuum(p, max_window=CONTINUUM_MAX_WINDOW, smooth_sigma=CONTINUUM_SMOOTH_SIGMA,
                       chunk=2048):
    """Band-free continuum = smoothed upper envelope of p (bands are dips).

    p: (N, L) or (L,). Processed in row-chunks to keep peak memory ~chunk×L.
    """
    p = np.asarray(p, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    out = np.empty_like(p)
    for lo in range(0, p.shape[0], chunk):
        blk = p[lo:lo + chunk]
        if _HAVE_SCIPY:
            env = maximum_filter1d(blk, size=max_window, axis=-1, mode="nearest")
            cont = gaussian_filter1d(env, sigma=smooth_sigma, axis=-1, mode="nearest")
        else:
            cont = _envelope_numpy(blk, max_window, smooth_sigma)
        out[lo:lo + chunk] = np.maximum(cont, blk)
    return out[0] if single else out


def _envelope_numpy(p, max_window, smooth_sigma):
    L = p.shape[-1]
    half = max_window // 2
    env = np.empty_like(p)
    for i in range(L):
        env[:, i] = p[:, max(0, i - half):min(L, i + half + 1)].max(axis=1)
    k = max(1, int(smooth_sigma))
    ker = np.ones(k) / k
    sm = np.empty_like(env)
    for r in range(env.shape[0]):
        pad = np.pad(env[r], k, mode="edge")
        sm[r] = np.convolve(pad, ker, mode="same")[k:-k]
    return sm


def grey(p, c, f=0.7, b=0.5):
    """The TRAINING family (reference). f∈[0,1] muting, b brightening."""
    p_mute = p + f * (c - p)
    return p_mute * (1.0 + b * f)


def non_grey(p, c, wl, f0=0.7, alpha=2.0, b0=0.5, lam0=1.0):
    """Wavelength-dependent muting f(λ)=f₀(λ/λ₀)^−α (blue-muted scattering haze).

    α=0 recovers grey; α>0 mutes the blue end more than the red — a family the
    grey-trained model never saw. b tilts with the same slope.
    """
    tilt = (np.asarray(wl) / lam0) ** (-alpha)
    f = np.clip(f0 * tilt, 0.0, 1.0)[None, :]
    b = (b0 * tilt)[None, :]
    p_mute = p + f * (c - p)
    return p_mute * (1.0 + b * f)


def band_selective(p, c, f=0.7, gamma=1.5, b=0.5):
    """Cloud-top-pressure proxy: strong (deep) band cores mute LESS than wings.

    Weight muting by (1 − (d/max d)^γ) so saturated cores — whose optical depth
    peaks above the cloud — are preserved relative to shallow wings. Breaks the
    "all bands mute by the same fraction" shortcut a network can exploit.
    """
    d = np.clip(c - p, 0.0, None)
    dmax = np.maximum(d.max(axis=-1, keepdims=True), 1e-300)
    w = 1.0 - (d / dmax) ** gamma        # deep cores → small weight → mute less
    p_mute = p + f * w * (c - p)
    return p_mute * (1.0 + b * f)


def patchy(p, c, phi=0.6, f=0.9, b=0.5):
    """Fractional cloud cover φ: φ·cloudy + (1−φ)·clear (Earth is φ≈0.6).

    The dominant real-world effect and the one the φ=1 training family omits
    entirely — the realtest Earth target is itself a patchy-cloud planet.
    """
    cloudy = grey(p, c, f=f, b=b)
    return phi * cloudy + (1.0 - phi) * p


def apply_family(name, p, c, wl):
    """Dispatch to a family using its documented representative parameters.

    p, c: (N, L) planet flux + continuum. wl: (L,) wavelength grid (µm).
    """
    pr = FAMILY_PARAMS[name]
    if name == "grey":
        return grey(p, c, **pr)
    if name == "non_grey":
        return non_grey(p, c, wl, **pr)
    if name == "band_selective":
        return band_selective(p, c, **pr)
    if name == "patchy":
        return patchy(p, c, **pr)
    raise ValueError(f"unknown cloud family {name!r}")


HELD_OUT_FAMILIES = ("non_grey", "band_selective", "patchy")
