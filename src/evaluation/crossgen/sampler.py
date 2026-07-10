"""Sample evaluation planets by bootstrapping real INARA rows from summary.csv.

Drawing actual INARA planets (composition + physical params) means the ONLY
thing that differs from the INARA training/eval set is the generator physics
(TauREx reflected-light proxy vs NASA PSG). This removes the label-distribution
confound: the gas-abundance marginals and their inter-species correlations match
INARA exactly. Each planet carries its real 12-species target vector for scoring.
"""
import os
import numpy as np
import pandas as pd

REPO_ROOT = "/Users/aakashrajput/MachineLearning/Exoplanets"
SUMMARY_PATH = os.path.join(REPO_ROOT, "data", "summary.csv")

# INARA's 12 target species, in the fixed order the models output.
INARA_TARGETS = [
    "H2O", "CO2", "O2", "N2", "CH4", "N2O",
    "CO", "O3", "SO2", "NH3", "C2H6", "NO2",
]

# Species TauREx has bundled opacities for AND that overlap INARA's targets.
TAUREX_SPECIES = ["H2O", "CO2", "O2", "CH4", "O3"]

R_EARTH_KM = 6371.0
RHO_EARTH_GCC = 5.513  # Earth bulk density, g/cm^3


def sample_planets(n, seed=0, species=TAUREX_SPECIES, summary_path=SUMMARY_PATH):
    """Bootstrap n planets from real INARA rows.

    summary.csv columns used: star_class, star_temperature, star_radius (solar),
    distance_parsec, semimajor_axis (au), planet_radius (km), planet_density
    (g/cm3), surface_pressure (bar), surface_temperature (K), and the 12 species.
    star_mass / planet_mass aren't stored, so they're derived (rough MS relation /
    density*volume).
    """
    df = pd.read_csv(summary_path)
    rng = np.random.default_rng(seed)
    rows = df.iloc[rng.integers(0, len(df), size=n)].reset_index(drop=True)

    planets = []
    for _, r in rows.iterrows():
        comp = {s: float(max(r[s], 1e-12)) for s in species}
        total = sum(comp.values())
        if total >= 0.999:  # keep N2 fill positive
            comp = {k: v * (0.999 / total) for k, v in comp.items()}

        planet_radius_km = float(r["planet_radius"])
        # M/M_earth = (rho/rho_earth) * (R/R_earth)^3
        planet_mass_earth = (float(r["planet_density"]) / RHO_EARTH_GCC) * \
            (planet_radius_km / R_EARTH_KM) ** 3
        star_radius = float(r["star_radius"])

        planets.append({
            "star_temperature": float(r["star_temperature"]),
            "star_radius": star_radius,
            "star_mass": float(np.clip(star_radius ** 0.8, 0.3, 2.0)),  # rough MS
            "star_class": int(r["star_class"]),
            "distance_parsec": float(r["distance_parsec"]),
            "sma": float(r["semimajor_axis"]),
            "planet_radius_km": planet_radius_km,
            "planet_mass_earth": float(np.clip(planet_mass_earth, 0.05, 50.0)),
            "surface_temperature": float(r["surface_temperature"]),
            "surface_pressure_bar": float(r["surface_pressure"]),
            "composition": comp,                                  # linear, TauREx species
            "target_vector": r[INARA_TARGETS].to_numpy(dtype=float),  # real 12-vec
        })
    return planets


def composition_to_target_vector(comp):
    """Fallback 12-vector builder (bootstrap path uses planet['target_vector'])."""
    v = np.zeros(len(INARA_TARGETS), dtype=float)
    for s, val in comp.items():
        v[INARA_TARGETS.index(s)] = val
    v[INARA_TARGETS.index("N2")] = max(0.0, 1.0 - sum(comp.values()))
    return v
