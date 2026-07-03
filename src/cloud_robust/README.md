# Cloud-Robust Retraining (Option 2)

Builds **cloud-robust** versions of all five model tracks by augmenting training
with parametrically-clouded INARA spectra, and quantifies the gain vs. the
original clear-trained models.

## Why
INARA's PSG spectra are **cloud-free**. In reflected light (`planet_signal`,
0.2–2 µm) a cloud **mutes molecular bands** (light reflects off the cloud top,
seeing less gas) and **brightens the continuum**. A clear-trained model reads a
muted band as "low abundance" — the cloud–abundance degeneracy — so cloudy
spectra are out-of-distribution. Here we teach each model to see through clouds.

## Isolation guarantee
This folder is fully self-contained. It only ever **reads** the original caches
(`data/cache`, `cache_original`, `cache_planet`), the original checkpoints, and
the original `model.py` files. It writes only under `src/cloud_robust/` and
`data/cache_cloudaug_*`. The original trained models are never modified.

## Cloud model (v1: grey + uniform)
See `cloud_model.py`. For a clear spectrum `p` with continuum `c` (upper
envelope) and band depth `d = c − p`:

```
p_mute  = p + f·(c − p)        # f∈[0,1] cloud opacity: mutes bands (f=1 → c)
p_cloud = p_mute · (1 + b·f)   # b≥0 continuum brightening
```

Grey (`f`, `b` wavelength-independent) and uniform (all bands muted equally).
For the 2-channel cache, only the `planet_signal` channel is clouded; the
star-dominated `star_planet_signal` channel is unchanged (clouds are invisible
at that scale). Wavelength-dependent / cloud-top-pressure refinements → v2.

## Files
| file | role |
|---|---|
| `cloud_model.py` | grey-cloud transform + continuum estimate + `python cloud_model.py` self-test |
| `build_cloud_cache.py` | precompute continuum + augmented stats per base cache |
| `cloud_dataset.py` | on-the-fly cloud augmentation (clear + 2 cloudy/epoch, fresh draws) |
| `train_common.py` | shared training loop + per-track registry |
| `<track>/train.py` | thin per-track wrapper → `src/cloud_robust/<track>/checkpoints/` |
| `eval_cloud_robustness.py` | 2×2 (model × clear/cloudy) + degradation curve |

Tracks: `original1dcnn`, `optimized1dcnn`, `2channel1dcnn`, `transformerarch`,
`causal`. The **causal** track concatenates cloud-augmented spectra onto the
existing env-counterfactual set (`train_x_augmented.pt`); its DSCM/counterfactual
generator is left untouched.

## Run order (in your terminal)
```bash
cd /Users/aakashrajput/MachineLearning/Exoplanets

# 0. (one-off) verify the cloud transform on one real spectrum — no training
python src/cloud_robust/cloud_model.py        # writes cloud_model_selftest.png

# 1. build the three augmentation caches (read-only on base caches)
python src/cloud_robust/build_cloud_cache.py             # all three
#   or: python src/cloud_robust/build_cloud_cache.py original

# 2. sanity-check a track for one epoch on a subset
python src/cloud_robust/original1dcnn/train.py --smoke

# 3. full 50-epoch runs (resumable; checkpoints under each <track>/checkpoints)
python src/cloud_robust/original1dcnn/train.py
python src/cloud_robust/optimized1dcnn/train.py
python src/cloud_robust/2channel1dcnn/train.py
python src/cloud_robust/transformerarch/train.py     # slow (batch 16)
python src/cloud_robust/causal/train.py              # slow (batch 16)

# 4. evaluate robustness (reads original + robust checkpoints, read-only)
python src/cloud_robust/eval_cloud_robustness.py
```

Notes:
- Fast tracks use batch 1024; `transformerarch`/`causal` use batch 16 and are the
  long poles. Training is resumable — re-running continues from the last checkpoint.
- `--num-workers N` on any `train.py` can speed up the on-the-fly augmentation.
- **Success criterion:** in `eval`, the original model's R² drops sharply
  clear→cloudy while the cloud-robust model recovers most of it on cloudy and
  stays comparable on clear.
