# Unified evaluation pipeline (`src/evaluation/`)

One offline-first driver that scores a trained checkpoint through **every applicable
evaluation**, organized into the sections the exoplanet-retrieval-ML literature uses, and
emits a single consolidated report + a cross-track leaderboard. Every number goes through
the **shared v2 core** (`core.py`) — the same leak-free observable `[contrast, stellar-SNR]`,
INARA-train-fit per-λ asinh norm, and CLR label decode used in training — so sections are
mutually consistent and consistent with `common/evaluate.py`.

## Why this shape

The model is a **reflected-light / thermal, 0.2–2.0 µm, disk-integrated, direct-imaging**
retrieval (LUVOIR-like) outputting a 12-species VMR **simplex**. That domain decides what
"real data" is even admissible: most ML-retrieval work is on **transiting** planets
(transmission/emission IR of hot Jupiters) — a *different observable*. So the only
**literal-ground-truth real** tests are **Solar-System-as-exoplanet** (E) and **real Earth**
(F); transiting-planet data (G, and the "far" rows of H) is a **wrong-observable OOD probe,
not accuracy**. Each section prints its epistemic status.

## Sections

| | Section | Ground truth | Credible source |
|---|---|---|---|
| A | In-distribution (INARA held-out test) | exact synthetic | reuses `common/evaluate.py` |
| B | Classical baselines (PriorMean/Ridge/RF) | exact | reuses `common/baselines.py` |
| C | Cross-generator (pRT / TauREx / MultiREx) | synthetic, diff. generator | local engines |
| D | PSG sanity anchor (eval-path control) | exact (real held-out PSG) | `data/inara_1by3` (0 API calls) |
| E | **Solar-System-as-exoplanet** | **literal** | NASA fact sheets; Lodders & Fegley 1998 |
| F | **Real disk-integrated Earth** | **literal, real photons** | Robinson+2011 VPL (EPOXI/earthshine-validated) |
| G | Transiting OOD probe | none (probe) | offline demo + optional real drop-ins |
| H | Published-retrieval comparison | pseudo-truth | literature retrievals + DOIs (bundled JSON) |
| I | Calibration (SBC / TARP / PIT / ECE) | exact | Talts+2020; Lemos+2023; Vasist+2023 |
| J | OOD honesty (δ/v, raw vs debiased R²) | — | `junk/VALIDATION_PLAN.md` B6 |

## Run

```bash
# everything applicable for a checkpoint (skips sections whose inputs are absent)
PYTHONPATH=src python3 -m evaluation.run_eval transformerarch --seeds 0 1 2 --suites all

# a subset by section letter or slug
PYTHONPATH=src python3 -m evaluation.run_eval optimized1dcnn --suites A C E F I
PYTHONPATH=src python3 -m evaluation.run_eval transformerarch --suites in_distribution calibration

# a checkpoint-suffix variant (e.g. the grey cloud model)
PYTHONPATH=src python3 -m evaluation.run_eval transformerarch --suffix _grey --suites A E F
```

Outputs → `src/evaluation/results/<track>/`: `report.md` (human), `summary.json` (roll-up),
per-section `<suite>/result.json` + plots; and `results/leaderboard.json` (accumulates across
tracks). A missing input is recorded as **skipped**, never a crash.

## Offline / credible-source design

- **Fully offline core.** Local generators + the bundled VPL Earth spectrum + an in-env
  band-template reflected engine (`engines/reflected_engine.py`, a transparent PROXY) +
  bundled cited benchmark tables (`benchmarks/*.json`). No network at runtime.
- **Optional online extras** live only in `fetch_benchmarks.py` (never imported by the core):
  populate `data/real_spectra/` (Section G) or `data/real_earth/` (Section F) when online.
- **Heavy compute is the user's.** Training seeds and large engine caches are generated
  separately (per project convention). To enable the gated sections:
  ```bash
  # D — PSG anchor (offline; gates cross-gen)
  python src/evaluation/crossgen/build_cache_v2_psg.py --out data/cache_v2_psg --n 2000
  # C — cross-gen caches (in the engine venv, then torch env)
  MultiREx-public/.venv/bin/python src/evaluation/crossgen/build_eval_cache.py \
      --engine {taurex|multirex} --feature-mode both --out data/cache_crossgen_<eng>_both
  python src/evaluation/crossgen/build_cache_v2_prt.py --prt-both data/cache_crossgen_<eng>_both \
      --out data/cache_v2_<eng>
  ```

## Extending

Add a section = one module in `suites/` exposing `SECTION`, `SUITE`, `TITLE`, `EPISTEMIC`,
`applicable(ctx)`, `run(ctx, **kw) -> core.SuiteResult`; register it in `registry.SUITES`.
Reuse `core.predict_cache` (cache dirs) / `core.predict_raw` (in-memory spectra) so scoring
stays identical, and `metrics_extra` for calibration/known-truth/honesty metrics.
```
