"""Transparent band-template reflected-light engine (pure numpy, no RT venv).

WHY THIS EXISTS. The pRT / TauREx engines do proper radiative transfer but need
their own heavy venvs (`.venv_prt`, `MultiREx-public/.venv`) and a separate
generation step. This engine is a *proxy* that runs IN the torch eval env, so the
solar-system known-truth (Section E) and published-target (Section H) suites work
fully offline in one process. It is deliberately simple and legible:

    reflected(λ) = B_λ(T_star) · albedo(λ) · exp( − Σ_s τ_s(λ) )
    τ_s(λ)       = OPACITY · g(VMR_s) · Σ_bands strength · Gaussian(λ; center, fwhm)
    g(VMR)       = log10(1 + VMR / VMR_REF)         # 0 for absent gas, grows with abundance

Band centers/strengths are literature molecular-band positions in 0.2–2.0 µm; a
Rayleigh ∝ λ⁻⁴ term brightens the blue continuum. This is a BAND-TEMPLATE PROXY,
not line-by-line RT: use it to test whether the model's prediction RESPONDS to a
molecule's presence/absence and abundance (monotonically, by construction), not for
absolute spectral fidelity. For RT fidelity, generate a pRT/TauREx cache and score
that through the same suite.

Interface matches evaluation/engines/{prt,taurex}_engine.py: shapes in arbitrary
units, absolute scale fixed downstream by calibration to INARA
(crossgen_eval/observable.calibration_factor).
"""
from __future__ import annotations

import numpy as np

PROXY = True

# cgs constants (Planck)
_H = 6.62607015e-27
_C = 2.99792458e10
_KB = 1.380649e-16

OPACITY = 0.9            # global band-depth scale
VMR_REF = 1e-8           # abundance floor above which a gas starts imprinting
ALBEDO0 = 0.3            # grey geometric-albedo baseline
RAYLEIGH_AMP = 0.6       # blue-continuum brightening amplitude
_RAY_REF_UM = 0.5

# Literature molecular band positions in 0.2-2.0 µm: species -> [(center_um, fwhm_um, rel_strength)].
BAND_CATALOG = {
    "H2O": [(0.72, 0.02, 0.3), (0.82, 0.03, 0.5), (0.94, 0.04, 0.7),
            (1.13, 0.05, 0.9), (1.38, 0.08, 1.0), (1.87, 0.10, 1.0)],
    "O2":  [(0.628, 0.006, 0.2), (0.688, 0.006, 0.4), (0.762, 0.008, 1.0), (1.27, 0.02, 0.5)],
    "O3":  [(0.30, 0.05, 1.0), (0.45, 0.15, 0.3), (0.60, 0.12, 0.6)],
    "CO2": [(1.44, 0.03, 0.2), (1.60, 0.04, 0.3), (2.00, 0.06, 0.6)],
    "CH4": [(0.62, 0.02, 0.2), (0.73, 0.03, 0.4), (0.79, 0.02, 0.3), (0.89, 0.03, 0.5),
            (1.00, 0.04, 0.6), (1.15, 0.05, 0.7), (1.40, 0.06, 0.5), (1.70, 0.08, 0.9)],
    "N2O": [(1.52, 0.03, 0.2), (1.96, 0.04, 0.3)],
    "CO":  [(1.57, 0.02, 0.15), (2.00, 0.03, 0.3)],
    "SO2": [(0.30, 0.05, 1.0), (0.34, 0.04, 0.5)],
    "NH3": [(0.55, 0.03, 0.2), (0.65, 0.03, 0.3), (0.93, 0.04, 0.4), (1.50, 0.06, 0.6), (2.00, 0.06, 0.5)],
    "C2H6": [(1.68, 0.05, 0.5), (2.00, 0.05, 0.4)],
    "NO2": [(0.40, 0.06, 0.6), (0.50, 0.08, 0.4)],
    "N2":  [],   # no bands in this range (Rayleigh scatterer only)
}

# Species this engine can imprint (has a non-empty band template) ∩ well-placed in 0.2-2.0 µm.
IMPRINTED_SPECIES = [s for s, b in BAND_CATALOG.items() if b]


def _planck_lambda(T, wl_um):
    wl_cm = np.asarray(wl_um, dtype=float) * 1e-4
    x = _H * _C / (wl_cm * _KB * float(T))
    return (1.0 / wl_cm ** 5) / np.expm1(x)


def _g(vmr):
    """Monotone abundance response: 0 for absent gas, ~n decades above VMR_REF."""
    return np.log10(1.0 + max(float(vmr), 0.0) / VMR_REF)


def _tau(comp, wl):
    tau = np.zeros_like(np.asarray(wl, dtype=float))
    for s, vmr in comp.items():
        gs = _g(vmr)
        if gs <= 0:
            continue
        for c, fwhm, strength in BAND_CATALOG.get(s, []):
            sigma = fwhm / 2.3548
            tau += OPACITY * gs * strength * np.exp(-0.5 * ((wl - c) / sigma) ** 2)
    return tau


def _albedo(wl):
    ray = 1.0 + RAYLEIGH_AMP * (_RAY_REF_UM / np.clip(wl, 1e-3, None)) ** 4
    return ALBEDO0 * np.clip(ray, 1.0, 8.0)


def generate_planet_shape(p, wl_grid_um):
    """Reflected-light planet spectrum SHAPE (arbitrary units) on wl_grid_um.

    Molecular bands darken a stellar × albedo continuum; deeper for more abundant
    gases (via _g). Absolute scale set downstream by calibration to INARA.
    """
    wl = np.asarray(wl_grid_um, dtype=float)
    continuum = _planck_lambda(p["star_temperature"], wl) * _albedo(wl)
    reflected = continuum * np.exp(-_tau(p["composition"], wl))
    return np.nan_to_num(reflected)


def stellar_continuum_shape(p, wl_grid_um):
    """Bare stellar continuum SHAPE (for the star+planet channel)."""
    return np.nan_to_num(_planck_lambda(p["star_temperature"], wl_grid_um))


def transmission_shape(p, wl_grid_um):
    """WRONG-observable analog: a transmission-style (1 − band absorption) spectrum,
    for the Section-G OOD probe. Not calibrated to INARA; deliberately off-distribution
    so the probe can demonstrate the model's behaviour on a transiting-type input."""
    wl = np.asarray(wl_grid_um, dtype=float)
    depth = 1.0 - np.exp(-_tau(p["composition"], wl))
    return np.nan_to_num(1.0 - 0.05 * depth / (depth.max() + 1e-30))
