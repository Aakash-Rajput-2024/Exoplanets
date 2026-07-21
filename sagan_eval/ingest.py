"""Ingest the Carl Sagan Institute solar-system catalog onto the INARA grid.

Source: Madden & Kaltenegger (2018), Astrobiology 18(12) 1559; data DOI 10.5281/zenodo.3930987
(CC-BY 4.0). 19 bodies, geometric albedo Ag(lambda), 0.45-2.5 um.

These are REAL measured reflectance spectra, unlike evaluation/engines/reflected_engine.py
which synthesises band templates. That is the whole point of using them.

Three facts drive every choice below, all verified against the files:

  1. The catalog is internally consistent: Spec_Sun / Albedo recovers the SAME solar SED
     for every body (frac diff 2e-5 Moon vs Earth). So Ag is the only per-body quantity.

  2. The Lundock ground-based spectra blow up in the deep telluric H2O bands
     (1.36-1.41 and 1.87-1.95 um), where the earthshine/moonshine ratio is ~0/0.
     Neptune spans -47..+133; physical geometric albedo is ~[0, 1.5]. Those points are
     masked and interpolated over, NOT clipped -- clipping would invent a flat band.

  3. The catalog's blue cutoff is 0.45 um but the model's input grid starts at 0.2 um
     (35% of the 4379 bins). Nothing can be recovered there. We flat-extrapolate and
     record ``blue_fill_frac`` per body; ``bluegap.py`` measures what that costs.

Read-only. Writes nothing outside sagan_eval/.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CATALOG = os.path.join(REPO, "data", "sagan_catalog", "CatalogofSolarSystemObjects")
ALBEDO_DIR = os.path.join(CATALOG, "Albedos")
SPEC_SUN_DIR = os.path.join(CATALOG, "Spectra", "NativeResolution", "Sun")

sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "evaluation", "crossgen"))

AG_MIN, AG_MAX = 0.0, 1.5      # physical geometric-albedo range (Enceladus ~1.0 is the max real)
SUN_TEFF = 5772.0              # matches star_sun in benchmarks/solar_system_truth.json


# --------------------------------------------------------------------------- #
# Canonical observation per body.
#
# Where the catalog has repeats we take the LOWEST unphysical-point fraction, with one
# physical override: Saturn's "withRing" spectra mix the ring system's reflectance into
# the disk-integrated albedo, and rings are not an atmosphere. We use the ring-free
# north-pole observation (which is also the cleanest, 0.0% bad).
# --------------------------------------------------------------------------- #
CANONICAL = {
    "Earth":     "Earth_Lundock081121",
    "Venus":     "Venus_MeadowsSpencer",
    "Mars":      "Mars_McCord1971",
    "Titan":     "Titan_Lundock080505",
    "Jupiter":   "Jupiter_Lundock080507",
    "Saturn":    "Saturn_Lundock081119NPole",     # ring-free, 0.0% bad
    "Uranus":    "Uranus_Lundock081120",
    "Neptune":   "Neptune_Lundock081120",
    "Mercury":   "Mercury_Mallama2017",
    "Moon":      "Moon_Lundock081121Luna1",
    "Io":        "Io_Fanale1974",
    "Europa":    "Europa_Spencer1999trailing",
    "Ganymede":  "Ganymede_Spencer1999leading",
    "Callisto":  "Callisto_Spencer1999leading",
    "Enceladus": "Enceladus_VIMS",
    "Dione":     "Dione_Lundock080505",
    "Rhea":      "Rhea_Lundock081125",
    "Ceres":     "Ceres_Lundock081125",
    "Pluto":     "Pluto_LorenziProtopapa",
}

# How each body relates to the model's 12-species reflected-light retrieval.
#   terrestrial : substantial atmosphere inside the 12-simplex -> ACCURACY test
#   giant       : H2/He dominated, mass sits outside the simplex -> HONESTY test
#   airless     : no atmosphere at all; spectrum is pure surface -> FALSE-POSITIVE test
BODY_CLASS = {
    "Earth": "terrestrial", "Venus": "terrestrial", "Mars": "terrestrial",
    "Titan": "terrestrial",
    "Jupiter": "giant", "Saturn": "giant", "Uranus": "giant", "Neptune": "giant",
    "Mercury": "airless", "Moon": "airless", "Io": "airless", "Europa": "airless",
    "Ganymede": "airless", "Callisto": "airless", "Enceladus": "airless",
    "Dione": "airless", "Rhea": "airless", "Ceres": "airless",
    "Pluto": "airless",   # 1e-5 bar N2/CH4; negligible for reflected-light band depth
}

# Bodies whose reflected spectrum is set by a cloud/haze deck rather than the clear
# gas column. These are the declouder's targets (Section D of run_sagan.py).
CLOUDY = {"Venus": "H2SO4 deck", "Titan": "organic haze", "Jupiter": "NH3 clouds",
          "Saturn": "NH3 clouds", "Earth": "H2O clouds (~60% cover)"}


def _planck_lambda(T, wl_um):
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = np.asarray(wl_um, dtype=float) * 1e-6
    return (2 * h * c ** 2 / lam ** 5) / (np.expm1(h * c / (lam * kB * T)))


def inara_grid():
    from inara_grid import load_inara_grid
    return np.asarray(load_inara_grid(), dtype=float)


def _read(stem, kind="albedo"):
    path = (os.path.join(ALBEDO_DIR, f"{stem}_Albedo.txt") if kind == "albedo"
            else os.path.join(SPEC_SUN_DIR, f"{stem}_Spec_Sun_HiRes.txt"))
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def _spike_mask(ag, valid=None, win=65, k=6.0, min_n=60):
    """Mask points far from a LOCAL running median, in local robust (MAD) units.

    The MAD must be local: photon noise in these ground-based spectra grows ~10-100x
    from 0.5 um to 2.0 um (Earth: sigma 0.010 -> 0.113), and Titan's albedo sits near
    zero across most of the NIR (deep CH4 bands). A single global MAD therefore flags
    ~9% of every high-res body and 45% of Titan -- real structure, not spikes.

    A real absorption band is many contiguous bins deep, so the local median tracks it;
    a 0/0 blowup is 1-3 bins tall and departs from its own neighbourhood.

    Bodies with < ``min_n`` points are published composite albedos (Venus, Mars, Io,
    ...), already clean and too sparse for a rolling statistic -- left untouched.
    """
    n = len(ag)
    if n < min_n:
        return np.zeros(n, bool)
    valid = np.ones(n, bool) if valid is None else valid
    win = min(win, (n // 4) * 2 + 1)
    half = win // 2
    out = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = ag[lo:hi][valid[lo:hi]]
        if w.size < 5:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad <= 0:
            continue
        out[i] = abs(ag[i] - med) > k * 1.4826 * mad
    return out


def clean_albedo(wl, ag):
    """Mask unphysical + spike points, interpolate across them in log-lambda.

    Interpolate, never clip: clipping a blown-up telluric band to AG_MAX would invent a
    bright flat feature exactly where the data says nothing.
    """
    bad = (~np.isfinite(ag)) | (ag < AG_MIN) | (ag > AG_MAX)
    bad |= _spike_mask(np.nan_to_num(ag), valid=~bad)
    if bad.all():
        raise ValueError("all points masked")
    good = ~bad
    ag_fixed = ag.copy()
    ag_fixed[bad] = np.interp(np.log(wl[bad]), np.log(wl[good]), ag[good])
    return np.clip(ag_fixed, AG_MIN, AG_MAX), bad


def albedo_on_grid(stem, grid=None):
    """Cleaned Ag(lambda) resampled onto the full INARA grid.

    Blueward of the body's own coverage the albedo is held FLAT at its bluest measured
    value (``blue_fill_frac`` of the grid). Redward likewise (rarely triggered: only if a
    body stops before 2.0 um). Returns (ag_grid, meta).
    """
    grid = inara_grid() if grid is None else grid
    wl, ag = _read(stem, "albedo")
    ag_clean, bad = clean_albedo(wl, ag)

    lo, hi = wl.min(), wl.max()
    in_native = (grid >= lo) & (grid <= hi)
    ag_grid = np.interp(grid, wl, ag_clean)          # np.interp clamps outside -> flat fill
    n_in_window = ((wl >= 0.45) & (wl <= 2.0)).sum()

    meta = dict(
        stem=stem,
        n_native=int(len(wl)),
        n_native_in_window=int(n_in_window),
        native_lo_um=float(lo), native_hi_um=float(hi),
        masked_frac=float(bad.mean()),
        blue_fill_frac=float((grid < lo).mean()),
        red_fill_frac=float((grid > hi).mean()),
        grid_covered_frac=float(in_native.mean()),
        # effective resolution of the native sampling, at 1 um
        native_R_at_1um=float(1.0 / np.median(np.diff(wl))) if len(wl) > 1 else np.nan,
        ag_median=float(np.median(ag_clean)),
    )
    return ag_grid, meta


def solar_sed_on_grid(grid=None, ref_stem="Moon_Lundock081121Luna1"):
    """Solar SED on the INARA grid.

    Derived from the catalog itself (Spec_Sun / Albedo) where the catalog reaches, and
    continued below its blue cutoff by a Planck(5772 K) curve scaled to match at the
    seam. Verified: Planck agrees with the catalog SED to ~8% median over 0.45-2.0 um.

    NOTE this only sets the model's SNR channel. The contrast channel is
    Fp/(Fstar+Fp) with Fp = Ag*Fsun and Fstar = Fsun, so Fsun cancels exactly there
    (see build_obs.verify_sed_cancellation).
    """
    grid = inara_grid() if grid is None else grid
    wl, ag = _read(ref_stem, "albedo")
    _, spec = _read(ref_stem, "spec")
    with np.errstate(divide="ignore", invalid="ignore"):
        sed = spec / ag
    ok = np.isfinite(sed) & (sed > 0) & (ag > 1e-3)
    wl_s, sed_s = wl[ok], sed[ok]
    ok2 = ~_spike_mask(sed_s / _planck_lambda(SUN_TEFF, wl_s))
    wl_s, sed_s = wl_s[ok2], sed_s[ok2]

    seam = wl_s.min()
    out = np.interp(grid, wl_s, sed_s)
    blue = grid < seam
    if blue.any():
        pl = _planck_lambda(SUN_TEFF, grid)
        scale = np.interp(seam, wl_s, sed_s) / _planck_lambda(SUN_TEFF, seam)
        out[blue] = pl[blue] * scale
    return out


def build_all(bodies=None):
    """{body: (ag_grid, meta)} for the canonical observation of each body."""
    grid = inara_grid()
    bodies = list(CANONICAL) if bodies is None else bodies
    out = {}
    for b in bodies:
        ag, meta = albedo_on_grid(CANONICAL[b], grid)
        meta["body"] = b
        meta["body_class"] = BODY_CLASS[b]
        meta["cloudy"] = CLOUDY.get(b, "")
        out[b] = (ag, meta)
    return out


if __name__ == "__main__":
    grid = inara_grid()
    print(f"INARA grid: {len(grid)} pts, {grid.min():.3f}-{grid.max():.3f} um\n")
    hdr = (f"{'body':<10}{'class':<13}{'n_nat':>6}{'R@1um':>7}{'lo_um':>7}"
           f"{'masked%':>9}{'bluefill%':>10}{'covered%':>9}{'Ag_med':>8}")
    print(hdr); print("-" * len(hdr))
    for b, (ag, m) in build_all().items():
        print(f"{b:<10}{m['body_class']:<13}{m['n_native']:>6}{m['native_R_at_1um']:>7.0f}"
              f"{m['native_lo_um']:>7.3f}{100*m['masked_frac']:>9.2f}"
              f"{100*m['blue_fill_frac']:>10.1f}{100*m['grid_covered_frac']:>9.1f}{m['ag_median']:>8.3f}")
