"""Build a cross-generator evaluation cache.

Samples planets, generates emission spectra with the chosen engine, matches the
INARA observable, interpolates onto INARA's 4379-pt grid, and writes:
  - val_x.npy   [N, C, 4379]  raw flux (un-standardized)
  - val_y.npy   [N, 12]       linear abundances in INARA order
  - manifest.json             engine, feature_mode, N, covered species, grid len

Writes .npy (not .pt) so the generation venv (TauREx / petitRADTRANS) needs no
torch. The per-model crosstest.py loads these, applies INARA training stats, and
runs the model.

Run in the generation venv, e.g.:
  MultiREx-public/.venv/bin/python -m crossgen_eval.build_eval_cache \
      --engine taurex --feature-mode planet --n 50 \
      --out data/cache_crossgen_taurex_planet
"""
import os
import sys
import json
import argparse
import numpy as np

# Make sibling modules importable whether run as a module or a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inara_grid import load_inara_grid
from sampler import sample_planets, composition_to_target_vector, TAUREX_SPECIES
from observable import (calibration_factor, INARA_PLANET_MEDIAN,
                        INARA_STARPLANET_MEDIAN)


def build(engine, feature_mode, n, out, seed=0):
    wl = load_inara_grid()
    print(f"INARA grid: {len(wl)} points, {wl.min():.3f}-{wl.max():.3f} um")
    planets = sample_planets(n, seed=seed)

    if engine == "taurex":
        from engines.taurex_engine import generate_planet_shape, stellar_continuum_shape
    elif engine == "prt":
        from engines.prt_engine import generate_planet_shape, stellar_continuum_shape
    else:
        raise ValueError(f"unknown engine {engine}")

    # 1. Generate spectrum SHAPES (arbitrary units) per planet.
    planet_shapes, star_shapes, Y = [], [], []
    for i, p in enumerate(planets):
        planet_shapes.append(generate_planet_shape(p, wl))
        if feature_mode == "both":
            star_shapes.append(stellar_continuum_shape(p, wl))
        # bootstrap path carries the real INARA label vector; else derive it
        Y.append(p.get("target_vector", composition_to_target_vector(p["composition"])))
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  generated {i + 1}/{n}")
    planet_shapes = np.stack(planet_shapes)  # [N, L]

    # 2. Empirical per-channel calibration to INARA's absolute scale.
    k_planet = calibration_factor(planet_shapes, INARA_PLANET_MEDIAN)
    planet_signal = planet_shapes * k_planet
    calib = {"planet": k_planet}
    if feature_mode == "planet":
        channels = [planet_signal]
    elif feature_mode == "both":
        star_shapes = np.stack(star_shapes)
        k_star = calibration_factor(star_shapes, INARA_STARPLANET_MEDIAN)
        star_planet = star_shapes * k_star
        calib["star_planet"] = k_star
        channels = [star_planet, planet_signal]  # matches dataloader channel order
    else:
        raise ValueError(f"unknown feature_mode {feature_mode}")

    X = np.nan_to_num(np.stack(channels, axis=1)).astype(np.float32)  # [N, C, L]
    Y = np.stack(Y).astype(np.float32)                                # [N, 12]
    print(f"calibration factors: {calib}")

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "val_x.npy"), X)
    np.save(os.path.join(out, "val_y.npy"), Y)
    manifest = {
        "engine": engine,
        "feature_mode": feature_mode,
        "n": int(n),
        "seed": int(seed),
        "covered_species": list(TAUREX_SPECIES) + ["N2"],
        "wl_points": int(len(wl)),
        "x_shape": list(X.shape),
        "calibration": calib,
        "observable": "reflected_light",
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}  X={X.shape}  Y={Y.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["taurex", "prt"], default="taurex")
    ap.add_argument("--feature-mode", choices=["planet", "both"], default="planet")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.engine, a.feature_mode, a.n, a.out, a.seed)
