"""petitRADTRANS thermal-emission backend (matching INARA's planet-channel physics).

INARA's planet channel is dominated by thermal emission (not reflected starlight).
An earlier version used `scattering_in_emission=True` which produces reflected-
light spectra with a spectrally *inverted* contrast shape (high blue / Rayleigh
vs INARA's high red / thermal) — causing the cross-gen eval to see out-of-
distribution input and collapse to the prior mean.

Now uses emission-only (`scattering_in_emission=False`) to match the physics that
the v2 models were trained on. Same interface as taurex_engine:
generate_planet_shape / stellar_continuum_shape return spectrum SHAPES (arbitrary
units); absolute scale is fixed downstream by empirical calibration to INARA.

Runs in .venv_prt. The Radtrans object (opacity load is slow) is built once and
reused across planets.
"""
import numpy as np

# c-k species we have opacities for AND that overlap INARA's targets.
PRT_SPECIES = ["H2O", "CO2", "CH4", "O2", "O3"]
MOLAR_MASS = {"H2O": 18.015, "CO2": 44.01, "CH4": 16.04,
              "O2": 32.0, "O3": 48.0, "N2": 28.013}

# cgs constants
_G = 6.674e-8          # cm^3 g^-1 s^-2
_M_EARTH_G = 5.972e27
_R_SUN_CM = 6.957e10
_AU_CM = 1.495978707e13
_H = 6.62607015e-27
_C = 2.99792458e10
_KB = 1.380649e-16

_WL_MIN, _WL_MAX = 0.3, 2.0   # pRT c-k coverage floor is 0.3 um
_N_LAYERS = 40


_RT = None
_PRESSURES_BAR = np.logspace(-6, 1, _N_LAYERS)  # 1e-6 .. 10 bar


def _get_radtrans():
    global _RT
    if _RT is None:
        from petitRADTRANS.radtrans import Radtrans
        _RT = Radtrans(
            pressures=_PRESSURES_BAR,
            wavelength_boundaries=[_WL_MIN, _WL_MAX],
            line_species=list(PRT_SPECIES),
            rayleigh_species=["N2"],
            gas_continuum_contributors=["N2--N2"],
            line_opacity_mode='c-k',
            scattering_in_emission=False,
        )
    return _RT


def _planck_lambda(T, wl_um):
    wl_cm = np.asarray(wl_um, dtype=float) * 1e-4
    x = _H * _C / (wl_cm * _KB * float(T))
    return (1.0 / wl_cm ** 5) / np.expm1(x)


def _mass_fractions(comp):
    """Volume mixing ratios (+N2 fill) -> mass fractions + mean molar mass (amu)."""
    vmr = dict(comp)
    vmr["N2"] = max(1e-6, 1.0 - sum(comp.values()))
    mmw = sum(vmr[s] * MOLAR_MASS[s] for s in vmr)
    mf = {s: np.full(_N_LAYERS, vmr[s] * MOLAR_MASS[s] / mmw) for s in vmr}
    return mf, mmw


def generate_planet_shape(p, wl_grid_um):
    """Thermal-emission spectrum SHAPE on wl_grid_um (arbitrary units)."""
    rt = _get_radtrans()
    nfreq = len(rt.frequencies)

    # mild monotonic T-P gradient anchored at the surface temperature
    frac = np.linspace(0.45, 1.0, _N_LAYERS)  # cooler aloft -> warmer at depth
    temperatures = np.clip(p["surface_temperature"] * frac, 50.0, 1500.0)
    mass_fractions, mmw = _mass_fractions(p["composition"])

    g_cgs = _G * (p["planet_mass_earth"] * _M_EARTH_G) / (p["planet_radius_km"] * 1e5) ** 2
    wl_cm, flux, _ = rt.calculate_flux(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        mean_molar_masses=np.full(_N_LAYERS, mmw),
        reference_gravity=float(g_cgs),
    )
    wl_um = np.asarray(wl_cm) * 1e4
    flux = np.nan_to_num(np.asarray(flux))
    order = np.argsort(wl_um)
    # left/right edge fill for INARA grid points outside pRT's 0.3-2.0 um coverage
    return np.interp(wl_grid_um, wl_um[order], flux[order],
                     left=flux[order][0], right=flux[order][-1])


def stellar_continuum_shape(p, wl_grid_um):
    """Bare stellar continuum SHAPE (for the star+planet channel)."""
    return np.nan_to_num(_planck_lambda(p["star_temperature"], wl_grid_um))
