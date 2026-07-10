"""TauREx-based reflected-light spectrum backend for cross-generator evaluation.

INARA's `planet_signal` (PSG) in 0.2-2.0 um is dominated by *reflected starlight*
(continuum peaking in the visible, with molecular absorption bands), NOT thermal
emission. TauREx's EmissionModel only models thermal emission (a rising IR Wien
tail for a ~288 K planet), so its shape is wrong for this comparison.

To match INARA's observable while staying on the TauREx side, we build a
reflected-light spectrum:

    reflected(lambda) = B_lambda(T_star) * (1 - beta * depth_norm(lambda))

where B_lambda(T_star) is the stellar Planck continuum (sets the visible peak)
and depth_norm comes from TauREx's *transmission* spectrum (sets the molecular
absorption bands: where the atmosphere is opaque, reflected light is darker).
Absolute scale is fixed later by empirical calibration to INARA's distribution
(see build_eval_cache / observable). This is a simplified reflected-light model;
petitRADTRANS (deferred) would do proper scattering.

Runs in MultiREx-public/.venv (no torch needed).
"""
import numpy as np

# Importing multirex configures TauREx's opacity cache (bundled H2O/CO2/O2/CH4/O3)
# and gives us a convenient transmission model builder.
import multirex as mrex
from astropy.constants import R_earth

R_EARTH_KM = R_earth.to("km").value  # ~6371

# CGS physical constants for the Planck function (units cancel under calibration).
_H = 6.62607015e-27   # erg s
_C = 2.99792458e10    # cm / s
_KB = 1.380649e-16    # erg / K
_BETA = 0.7           # molecular-band imprint strength (0..1)


def _planck_lambda(T, wl_um):
    """Blackbody spectral radiance shape B_lambda(T) (arbitrary units)."""
    wl_cm = np.asarray(wl_um, dtype=float) * 1e-4
    x = _H * _C / (wl_cm * _KB * float(T))
    return (1.0 / wl_cm ** 5) / np.expm1(x)


def _transmission_depth(p, wl_grid_um):
    """TauREx transit depth (Rp/Rs)^2 on wl_grid_um, via MultiREx's builder."""
    star = mrex.Star(temperature=p["star_temperature"],
                     radius=p["star_radius"], mass=p["star_mass"])
    planet = mrex.Planet(radius=p["planet_radius_km"] / R_EARTH_KM,  # Earth radii
                         mass=p["planet_mass_earth"])                # Earth masses
    atm = mrex.Atmosphere(
        temperature=p["surface_temperature"],
        base_pressure=p["surface_pressure_bar"] * 1e5,  # bar -> Pa
        top_pressure=1e-3,
        fill_gas="N2",
        composition={g: float(np.log10(v)) for g, v in p["composition"].items()},
    )
    planet.set_atmosphere(atm)
    system = mrex.System(star=star, planet=planet, sma=p["sma"])
    system.make_tm()
    wn = np.sort(10000.0 / np.asarray(wl_grid_um, dtype=float))
    bin_wn, depth = system.generate_spectrum(wn)
    wl_from_wn = 10000.0 / np.asarray(bin_wn)
    order = np.argsort(wl_from_wn)
    return np.interp(wl_grid_um, wl_from_wn[order], np.asarray(depth)[order])


def generate_planet_shape(p, wl_grid_um):
    """Reflected-light planet spectrum SHAPE on wl_grid_um (arbitrary units).

    Absolute scale is set downstream by empirical calibration to INARA.
    """
    depth = _transmission_depth(p, wl_grid_um)
    rng = depth.max() - depth.min()
    depth_norm = (depth - depth.min()) / (rng + 1e-30)           # 0..1
    continuum = _planck_lambda(p["star_temperature"], wl_grid_um)  # visible peak
    reflected = continuum * (1.0 - _BETA * depth_norm)             # imprint bands
    return np.nan_to_num(reflected)


def stellar_continuum_shape(p, wl_grid_um):
    """Bare stellar continuum SHAPE (for the star+planet channel)."""
    return np.nan_to_num(_planck_lambda(p["star_temperature"], wl_grid_um))
