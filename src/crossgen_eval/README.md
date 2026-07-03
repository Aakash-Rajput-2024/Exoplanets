# Cross-generator evaluation

Evaluate the INARA-trained models on synthetic spectra from a **different physics
generator**, to test whether they generalize beyond NASA PSG (the generator that
produced the INARA training data) or overfit to its quirks.

**Nothing here modifies the INARA training/testing code, caches, or stats** — it
reads INARA's stats read-only and writes separate `cache_crossgen_*` dirs and
`crossgen_*` reports.

## Pipeline
1. `build_eval_cache.py` (run in `MultiREx-public/.venv`, no torch needed)
   samples planets, generates spectra with an engine, calibrates to INARA's
   scale, and writes `val_x.npy [N,C,4379]` + `val_y.npy [N,12]` + `manifest.json`.
2. `<model>/crosstest.py` (run in the PyTorch env, e.g. `venvNLP`) loads that
   cache, standardizes with INARA **training** stats, runs the checkpoint, and
   writes `crossgen_details_<engine>.txt` + `crossgen_charts_<engine>/`.

## Run
```bash
# generate (in .venv)
MultiREx-public/.venv/bin/python src/crossgen_eval/build_eval_cache.py \
    --engine taurex --feature-mode planet --n 2000 \
    --out data/cache_crossgen_taurex_planet
# evaluate (in venvNLP)
venvNLP/bin/python src/transformerarch/crosstest.py --engine taurex
```
`--feature-mode both` (+ `cache_crossgen_taurex_both`) is required for the
2-channel CNN.

## Engine notes (TauREx, current)
INARA's `planet_signal` in 0.2-2.0 um is **reflected starlight** (visible peak +
molecular bands), not thermal emission. TauREx's EmissionModel is thermal-only
(wrong shape), so `engines/taurex_engine.py` builds a **reflected-light proxy**:
stellar Planck continuum imprinted with molecular bands from TauREx's
transmission opacity, then empirically calibrated to INARA's median scale.

This is a *simplified* reflected-light model (no multiple scattering, no surface
albedo spectrum, no proper thermal+reflected mix). petitRADTRANS (deferred) does
proper scattering and is the path to higher fidelity — drop in
`engines/prt_engine.py` behind the same `generate_planet_shape` /
`stellar_continuum_shape` interface and run with `--engine prt`.

## Interpreting results — caveats
Negative R² currently reflects a mix of: (a) spectral domain shift (the real
thing we want to measure), (b) **label-distribution shift** (the sampler's
abundance marginals differ from INARA's — fix by matching them for a cleaner
test), and (c) the reflected-light approximation. Treat current numbers as a
working pipeline + preliminary signal, not a final verdict.
