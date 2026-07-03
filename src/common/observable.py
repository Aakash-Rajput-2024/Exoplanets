"""Observable definition + physical noise model (fixes C1).

The pipeline previously consumed the isolated, noiseless ``planet_signal`` — a
simulator internal that no instrument can measure (the planet is ~1e5x fainter
than the star and, blueward of ~0.5 um, below the noise). We replace it with a
genuine reflected-light observable and inject the instrument noise.

OBSERVABLE (reflected light, 0.2-2.0 um)
    contrast  C(λ) = F_p / F_star ≈ planet_signal / star_planet_signal
Dimensionless; carries the geometric-albedo shape (molecular bands + Rayleigh +
clouds) and drops the absolute distance/luminosity scale the network otherwise
overfits (the latent cause of the H3/H4 cross-generator collapse).

NOISE MODEL
    After starlight subtraction the planet estimate is  F_p_hat = F_p + n,
    n ~ N(0, σ²),  σ = the INARA noise column. Hence the contrast noise is
        σ_C(λ) = σ(λ) / F_star(λ) ≈ noise / star_planet = 1 / SNR(λ).
    Training injects  C_noisy = C + (σ_C / α)·ε ,  ε~N(0,1).
    α is an SNR multiplier: α=1 reproduces the dataset noise; the eval sweep
    varies α to trace accuracy vs SNR (achieved SNR is reported, not assumed).

MODES (ablatable — "which observable gives the best results" is a paper figure)
    contrast_snr : [C, SNR]   (default; SNR channel = per-λ trustworthiness)
    contrast     : [C]
    flux         : [planet_signal]      (legacy/unobservable — ablation only)
    star_planet  : [star_planet_signal] (raw detector — honest-but-hard ablation)
"""

from __future__ import annotations

import torch

_TINY = 1e-30
SNR_CLAMP = (1e-3, 1e3)   # kills the 2.0 um boundary artifact (noise→0 → SNR→1e19)
OBSERVABLE_MODES = ("contrast_snr", "contrast", "flux", "star_planet")


def contrast(star_planet, planet):
    return planet / torch.clamp(star_planet, min=_TINY)


def snr(star_planet, noise):
    return torch.clamp(star_planet / torch.clamp(noise, min=_TINY), *SNR_CLAMP)


def contrast_sigma(star_planet, noise):
    """Per-λ 1σ contrast noise σ_C = noise / star_planet (= 1/SNR)."""
    return torch.clamp(noise, min=0.0) / torch.clamp(star_planet, min=_TINY)


def make_observable(x, noise, mode="contrast_snr"):
    """x:(N,2,L) ch0=star_planet ch1=planet ; noise:(N,L) -> observable (N,C,L)."""
    sp, pl = x[:, 0, :], x[:, 1, :]
    if mode == "flux":
        return pl.unsqueeze(1)
    if mode == "star_planet":
        return sp.unsqueeze(1)
    C = contrast(sp, pl)
    if mode == "contrast":
        return C.unsqueeze(1)
    if mode == "contrast_snr":
        return torch.stack([C, snr(sp, noise)], dim=1)
    raise ValueError(f"mode must be one of {OBSERVABLE_MODES}, got {mode!r}")


def inject_noise(x, noise, mode, alpha=1.0, generator=None):
    """Return a noisy observable. Noise is added to the CONTRAST channel in linear
    space (σ_C/α), where it is physical and may drive contrast negative — the
    downstream asinh encoding handles the sign. The SNR channel is left clean
    (an instrument reports its own error bars); `flux`/`star_planet` modes get
    additive σ directly.
    """
    sp, pl = x[:, 0, :], x[:, 1, :]
    if mode in ("flux", "star_planet"):
        base = pl if mode == "flux" else sp
        eps = torch.randn(base.shape, generator=generator, device=base.device)
        return (base + (torch.clamp(noise, min=0.0) / alpha) * eps).unsqueeze(1)
    C = contrast(sp, pl)
    sig = contrast_sigma(sp, noise) / max(alpha, 1e-12)
    C_noisy = C + sig * torch.randn(C.shape, generator=generator, device=C.device)
    if mode == "contrast":
        return C_noisy.unsqueeze(1)
    if mode == "contrast_snr":
        return torch.stack([C_noisy, snr(sp, noise)], dim=1)
    raise ValueError(f"mode must be one of {OBSERVABLE_MODES}, got {mode!r}")


def achieved_snr(star_planet, noise, alpha):
    """Median per-spectrum SNR actually realized at multiplier α (for reporting)."""
    s = torch.clamp(star_planet / torch.clamp(noise, min=_TINY), *SNR_CLAMP) * alpha
    return float(s.median())
