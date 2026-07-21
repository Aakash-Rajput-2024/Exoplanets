# Exoplanet Atmosphere Retrieval from Reflected-Light Spectra

Neural networks that read the atmospheric composition of a rocky exoplanet off its
reflected-light spectrum (0.2-2.0 micron, the wavelength range a LUVOIR-class
direct-imaging telescope would see). Models are trained on INARA, a synthetic
library of ~110,000 terrestrial planets generated with NASA's Planetary Spectrum
Generator, and evaluated both on held-out synthetic planets and on real telescope
spectra of 19 Solar System bodies observed as if they were exoplanets.

The write-up with all headline results and figures is [report/REPORT.tex](report/REPORT.tex)
(compile with `pdflatex REPORT.tex` from inside `report/`, or just read the `.tex`
source — it's plain text). This README is about running the code, not the findings.

## Repository layout

```
src/
  common/          shared pipeline: data loading, observable construction, noise
                   model, CLR label transform, training loop, metrics, registry
  models/          the six retrieval architectures (below)
  cloud_recovery/  standalone de-clouding front-end (inference-only)
  evaluation/      the unified evaluation harness (sections A-J) + cross-generator testing
  data/            placeholder for dataset/cache helpers

sagan_eval/        validation against 19 real Solar-System spectra (Madden &
                   Kaltenegger 2018 catalog) — the "does this work on real data" test

report/            the paper draft (REPORT.tex) and its figures
compute/           cluster (SLURM) submission scripts
notebooks/         early data-exploration notebooks
Lit_rew/           literature review PDFs
data/              INARA spectra + derived caches (not in git — see Data below)
junk/              earlier drafts, retired code, working notes — safe to ignore,
                   kept locally for provenance, not needed to run anything

run.sh             train + evaluate, interactive
eval.sh            evaluate existing checkpoints only (the full A-J report)
requirements.txt   dependencies for everything above
```

Each track under `src/models/<name>/` also carries some older standalone
`train.py` / `test.py` / `dataloader.py` files from before the shared `common/`
pipeline existed. They still run but are superseded — `common.train_runner` and
`common.evaluate` are the current path for every track, and are what `run.sh` /
`eval.sh` call. `registry.py` in `common/` is the single source of truth for
which `model.py` belongs to which track.

### What's not in git

`.gitignore` deliberately keeps these out of version control:

- **Bulk dataset** — raw INARA spectra + derived caches (`data/cache*/`,
  `data/inara_1by3/`, `data/summary.csv`, `**/cache/`, `*.pt`) — tens of GB,
  regenerable from NASA PSG (see [Data](#data) below). Small curated
  *real*-observation reference sets (`data/real_earth/`, `data/sagan_catalog/`,
  a few MB total) are the exception — they're tracked, since they're hard to
  re-obtain and are the project's only real ground truth.
- **Literature review** — `/Lit_rew/` (~66 MB of background-paper PDFs).
- **Training run output** — checkpoints (`*.pth`, `**/checkpoints/`),
  TensorBoard runs (`**/runs/`), and root-level run logs (`/logs/`).
- **Claude Code config** — `.claude/`, `CLAUDE.md`.
- **Everything else generated/local** — `venv/`, `__pycache__/`, `.DS_Store`,
  the third-party `MultiREx-public/` clone (obtain separately, has its own git
  history), and `junk/` (retired code/notes, kept locally for provenance only).

## Setup

Python 3.10+ (developed against 3.11). PyTorch runs on CUDA, Apple Silicon (MPS),
or CPU — no special build needed on any of the three.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers training, evaluation, and the real-data validation.
Cross-generator testing against TauREx/petitRADTRANS needs extra third-party
packages that live in their own environments — see
[src/evaluation/crossgen/README.md](src/evaluation/crossgen/README.md) if you need that
part specifically; it's not required for the core training/eval loop.

## Data

`data/` holds the raw INARA spectra (`summary.csv` + `inara_1by3/`, ~94 GB of CSVs)
and the derived caches built from them (`cache_v2/` etc.). None of this is in git —
it's too large, and it's excluded in `.gitignore`. (`data/real_earth/` and
`data/sagan_catalog/` are the exception — a few MB of real telescope/reflectance
spectra, tracked normally; see [What's not in git](#whats-not-in-git).) If you're
working on the same machine this was developed on, the big stuff is already
sitting in `data/` and there's nothing to do. If you're setting up elsewhere,
you'll need the raw INARA data copied in separately, then build the cache once:

```bash
PYTHONPATH=src python -m common.build_cache        # reads data/summary.csv + data/inara_1by3
                                                     # writes data/cache_v2/ (needs ~6-8 GB RAM)
```

Everything downstream (training, evaluation) reads only from `data/cache_v2/`, so
this is the one data-prep step and it only needs to run once.

## Running things

Everything below assumes you're in the repo root with the venv active. All
entry points set `PYTHONPATH=src` for you if you use the wrapper scripts;
if you call a module directly, export it yourself: `export PYTHONPATH=src`.

### Training

```bash
./run.sh
```

Interactive: it asks which track(s) to train, how many epochs, how many seeds,
and whether to include the grey-cloud variant. Non-interactive form for scripting:

```bash
NONINTERACTIVE=1 TRACKS="optimized1dcnn" EPOCHS=50 SEEDS="0 1 2" ./run.sh
```

The six tracks (see `src/common/registry.py` for exact configs):

| track | what it is |
|---|---|
| `original1dcnn`   | the original NASA-paper CNN architecture, scaled up (36M params) |
| `optimized1dcnn`  | CNN + BatchNorm + progressive kernels, the reference model (1.2M params) |
| `transformerarch` | CNN stem into a small transformer |
| `causal`          | transformer + do(environment) counterfactual **augmentation** |
| `causal_cfi`      | transformer + counterfactual **invariance** objective (a real training loss, not augmentation) |
| `causal_xl`       | ~2M-parameter version of the causal_cfi objective |

`run.sh` trains, then evaluates on the test split, runs a leakage sanity probe,
and checks held-out cloud-family transfer, all in one pass. Everything is logged
to `logs/run_<timestamp>/`. Each run is fresh by default; pass `--resume` to
`common.train_runner` directly (see below) if a job gets interrupted and you
want to continue from `checkpoints_v2/last_<track>_seed<N>.pth` instead of
restarting — that's how the cluster scripts in `compute/` run it.

A single track/seed directly, if you don't want the interactive wrapper:

```bash
PYTHONPATH=src python -m common.train_runner optimized1dcnn --seed 0 --epochs 50 --resume
```

### Evaluation

Two levels, both operate only on already-trained checkpoints (neither trains or
mutates weights):

```bash
# quick: in-distribution test-set score + accuracy-vs-exposure sweep for one track
PYTHONPATH=src python -m common.evaluate optimized1dcnn --seeds 0 1 2

# full: every applicable section (baselines, cross-generator, solar-system,
# real Earth, calibration, OOD honesty...) for every trained checkpoint
./eval.sh
```

`eval.sh` accepts `ONLY="track1 track2"` to restrict which tracks it scores, and
`SUITES="A E F"` to restrict which sections run — see the comment block at the
top of the script for the full list. Output goes to
`src/evaluation/results/<track>/report.md` per track, plus a cross-track
`src/evaluation/results/COMPARISON.md`. See
[src/evaluation/README.md](src/evaluation/README.md) for what each section (A-J) means and
why — reflected-light retrieval only has two sources of literal ground truth
(Solar System bodies and real Earth), and the harness is explicit about which
sections are measuring real accuracy versus just probing for failure modes
(OOD behavior, calibration, cross-generator transfer).

### Testing on real data

The synthetic test split above tells you how well a model does on more INARA-like
spectra. The real test is `sagan_eval/`: 19 real telescope spectra of Solar System
bodies (four planets/moons with atmospheres, four gas giants, eleven airless
bodies used as a false-positive trap), run through the frozen models exactly as
if they were unknown exoplanets.

```bash
PYTHONPATH=src python sagan_eval/run_sagan.py       # score every track against the catalog
PYTHONPATH=src python sagan_eval/make_report_figs.py  # regenerate report/figures/ from results
```

Results land in `sagan_eval/sagan_results.json` and `sagan_eval/RESULTS.md`. This
is where the methane-detection / oxygen-fabrication findings in the report come
from.

### Cloud recovery

A separate, isolated de-clouding front-end (`src/cloud_recovery/`) that restores a
clouded spectrum before handing it to a frozen retrieval model, rather than
retraining retrieval models on clouds directly. Self-contained instructions in
[src/cloud_recovery/README.md](src/cloud_recovery/README.md).

## Cluster (compute/)

`compute/` has the SLURM submission scripts used to train on a GPU cluster
(written for a PARAM Rudra-style setup — adjust the partition/module names for a
different cluster):

- `setup_env.sh` — one-time conda environment setup on a compute node
- `submit_all.sh` — submits one job per (track, seed)
- `submit_eval.sh` — evaluates everything once training jobs finish
- `slurm_train_causal.sh` — array job for the `causal_xl` track specifically

These assume the repo (code only — not the multi-GB data/cache) has been copied
to the cluster filesystem, e.g. via `rsync`, and expect to be run from the repo
root: `bash compute/submit_all.sh`, `sbatch compute/slurm_train_causal.sh`.

## Literature review

`Lit_rew/` has the background papers this project builds on, including the
INARA/NASA-PSG dataset paper and prior ML-retrieval work (ExoGAN, Waldmann et al.,
neural posterior estimation for retrieval, etc.), plus a `Causality/` subfolder
for the do-calculus / counterfactual-invariance side of the causal track. It's
local-only, not tracked in git (see [What's not in git](#whats-not-in-git)) —
if you're setting up fresh elsewhere, this directory just won't exist and
nothing in `src/` depends on it at runtime.
