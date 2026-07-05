"""Observable definition + physical noise model (fixes C1; closes the v2 leak).

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
        σ_C(λ) = σ(λ) / F_star(λ) ≈ noise / stellar = 1 / SNR_star(λ).
    Training injects  C_noisy = C + (σ_C / α)·ε ,  ε~N(0,1).
    α is an SNR multiplier: α=1 reproduces the dataset noise; the eval sweep
    varies α to trace accuracy vs SNR (achieved SNR is reported, not assumed).

NOISE UNITS — VALIDATED (Zorzan et al. 2025, ApJS 277:38 §3.1;
notes/noise_units_finding.md)
    The noise column is the real 1σ per-bin instrument noise of a 15 m
    LUVOIR-like space telescope at R=1900: at ≤2 µm a coronagraph suppresses
    the star by 10¹⁰ (IWA 2λ/D), read noise 1 e⁻/pix, dark 0.001 e⁻/s; noise
    applies to the combined star+planet signal. Nominal exposure = 8 hr for an
    Earth analog at 5 pc, scaled ∝ distance². (>2 µm has no coronagraph/noise
    sim — the 2 µm boundary bin trimmed by data.TRIM_TAIL_BINS is that
    documented discontinuity.)

    Because every noise term (photon/read/dark) gives SNR ∝ √t at fixed
    distance,  **α = √(t_exp / t_nominal)** — α is exposure time, not a fudge:
    α=1 → the nominal 8 hr-equivalent program; α=10 → 100× (~800 hr @ 5 pc).
    At α=1 the planet sits ~10³× below the per-bin noise (band-integrated
    detection SNR < 0.1), so retrieval at the literal nominal exposure is
    photon-starved to R² ≈ 0 BY PHYSICS — the INARA paper itself trained on
    noiseless spectra for this reason (§5). Headline results must therefore be
    the R²-vs-exposure sweep, never a single α=1 number.

THE SNR CHANNEL — leak-safe (v2 fix)
    The auxiliary "how trustworthy is each wavelength" channel is the STELLAR SNR
    ``α · F_star / noise``. Two properties matter:
      * Its numerator is F_star = star_planet − planet, i.e. the STAR ALONE. The
        earlier version used ``star_planet / noise`` whose numerator is
        F_star + F_p, so the full planetary spectrum leaked into the model
        noise-free (bypassing the injected contrast noise entirely — the model
        learned to read F_p from its own "error bars"). Removing F_p from the
        numerator closes that leak: this channel is now a function of (F_star,
        noise, α) only, carrying zero planet-discriminative signal by
        construction. F_star is legitimately known to an observer (stellar model
        / calibration), F_p is not.
      * It scales with α, so at high SNR the reported error bars actually shrink
        in lockstep with the injected contrast noise (they were previously stale,
        contradicting the contrast channel across the sweep).

MODES (ablatable — "which observable gives the best results" is a paper figure)
    contrast_snr : [C, SNR_star]  (default; SNR channel = per-λ trustworthiness)
    contrast     : [C]
    flux         : [planet_signal]      (legacy/unobservable — ablation only)
    star_planet  : [star_planet_signal] (raw detector — honest-but-hard ablation)
"""

from __future__ import annotations

import torch

_TINY = 1e-30
SNR_CLAMP = (1e-3, 1e3)   # kills boundary artifacts (noise→0 → SNR blowup)
OBSERVABLE_MODES = ("contrast_snr", "contrast", "flux", "star_planet")


def _alpha_floor(alpha):
    """Clamp α away from 0, working for a python float OR a per-sample (B,1)
    tensor (exposure-time augmentation passes the latter)."""
    if torch.is_tensor(alpha):
        return torch.clamp(alpha, min=1e-12)
    return max(alpha, 1e-12)


def stellar(star_planet, planet):
    """F_star = (F_star + F_p) − F_p, reconstructed from the two stored channels.

    Used wherever the *star alone* is needed (SNR/error-bar channel, σ_C) so no
    planetary signal can leak into the model noise-free. Clamped ≥ 0 for safety;
    physically star_planet ≥ planet always.
    """
    return torch.clamp(star_planet - planet, min=_TINY)


def contrast(star_planet, planet):
    return planet / torch.clamp(star_planet, min=_TINY)


def snr(star_planet, planet, noise, alpha=1.0):
    """Per-λ STELLAR SNR × α — the instrument's known sensitivity, carrying no F_p."""
    return torch.clamp(
        alpha * stellar(star_planet, planet) / torch.clamp(noise, min=_TINY),
        *SNR_CLAMP,
    )


def contrast_sigma(star_planet, planet, noise):
    """Per-λ 1σ contrast noise σ_C = noise / F_star (= 1/stellar SNR)."""
    return torch.clamp(noise, min=0.0) / stellar(star_planet, planet)


def make_observable(x, noise, mode="contrast_snr", alpha=1.0):
    """x:(N,2,L) ch0=star_planet ch1=planet ; noise:(N,L) -> observable (N,C,L).

    Noise-free (the val/test baseline / norm-fitting input). ``alpha`` only scales
    the SNR channel here so a norm fitted at α=1 stays consistent with eval α=1.
    """
    sp, pl = x[:, 0, :], x[:, 1, :]
    if mode == "flux":
        return pl.unsqueeze(1)
    if mode == "star_planet":
        return sp.unsqueeze(1)
    C = contrast(sp, pl)
    if mode == "contrast":
        return C.unsqueeze(1)
    if mode == "contrast_snr":
        return torch.stack([C, snr(sp, pl, noise, alpha)], dim=1)
    raise ValueError(f"mode must be one of {OBSERVABLE_MODES}, got {mode!r}")


def inject_noise(x, noise, mode, alpha=1.0, generator=None):
    """Return a noisy observable. Noise is added to the CONTRAST channel in linear
    space (σ_C/α), where it is physical and may drive contrast negative — the
    downstream asinh encoding handles the sign. The SNR channel is the (planet-
    free) stellar SNR × α — clean because an instrument reports its own error
    bars, but leak-free because its numerator is the star alone. `flux`/
    `star_planet` modes get additive σ directly.
    """
    sp, pl = x[:, 0, :], x[:, 1, :]
    if mode in ("flux", "star_planet"):
        base = pl if mode == "flux" else sp
        eps = torch.randn(base.shape, generator=generator, device=base.device)
        return (base + (torch.clamp(noise, min=0.0) / _alpha_floor(alpha)) * eps).unsqueeze(1)
    C = contrast(sp, pl)
    sig = contrast_sigma(sp, pl, noise) / _alpha_floor(alpha)
    C_noisy = C + sig * torch.randn(C.shape, generator=generator, device=C.device)
    if mode == "contrast":
        return C_noisy.unsqueeze(1)
    if mode == "contrast_snr":
        return torch.stack([C_noisy, snr(sp, pl, noise, alpha)], dim=1)
    raise ValueError(f"mode must be one of {OBSERVABLE_MODES}, got {mode!r}")


def achieved_snr(star_planet, planet, noise, alpha):
    """Median per-spectrum STELLAR SNR actually realized at multiplier α (for the
    sweep x-axis). Star-only numerator, matching the SNR channel definition."""
    s = torch.clamp(stellar(star_planet, planet) / torch.clamp(noise, min=_TINY), *SNR_CLAMP) * alpha
    return float(s.median())


def snr_summary(star_planet, planet, noise, alpha):
    """Physically interpretable SNR diagnostics at multiplier α (sweep axes).

    Returns a dict:
      exposure_x       — t_exp/t_nominal = α² (all noise terms give SNR ∝ √t, so
                         α² is literally the exposure-time multiple of the
                         validated 8 hr @ 5 pc nominal program; Zorzan+25 §3.1).
      snr_star_median  — median per-bin STELLAR SNR (α·F_star/noise), the
                         error-bar channel the model actually sees.
      snr_planet_band  — median per-spectrum band-integrated PLANET detection
                         SNR  α·√(Σ_λ (F_p/σ)²)  — the matched-filter statistic
                         that says whether the planet is detectable AT ALL
                         (<~5 ⇒ retrieval is information-starved regardless of
                         architecture).
    """
    sig = torch.clamp(noise, min=_TINY)
    s_star = torch.clamp(stellar(star_planet, planet) / sig, *SNR_CLAMP) * alpha
    s_band = alpha * torch.sqrt(((planet / sig) ** 2).sum(dim=-1))
    return dict(
        exposure_x=float(alpha) ** 2,
        snr_star_median=float(s_star.median()),
        snr_planet_band=float(s_band.median()),
    )
