# Spectral de-clouding front-end (`src/cloud_recovery/`)

A **standalone** supervised network that recovers a near-clear reflected-light
spectrum from a **clouded** one, so a *frozen, clear-trained* retrieval model can be
run on the restored spectrum — **without ever retraining the retrieval model on
clouds**. This is the "play with clouds, don't pollute the models" path.

## Why supervised (not a VAE)

The cloud transform in [`common/cloud_families.py`](../common/cloud_families.py) is a
**known forward operator**, so every clear INARA spectrum + a random cloud draw is an
**exact `(cloudy, clear)` training pair**. With paired data a supervised 1D U-Net
directly minimises reconstruction error against the true clear spectrum — no latent
ambiguity, no risk of a decoder inventing structure the way an *unpaired* VAE would.

**The declouder's output is used for inference only.** It is never fed back into
retrieval *training* — that would inject the network's prior as if it were data,
exactly the contamination this module avoids.

## Isolation guarantee

Reads only the shared pipeline (`common.*`) and `data/cache_v2` **read-only**. Writes
**solely** under `src/cloud_recovery/` (`checkpoints/`, `eval_out/`, `cache/`). It imports no
training entry point and mutates no existing module, cache, or checkpoint. Delete this
folder and the rest of the repo is byte-for-byte unchanged.

## What it learns

Both sides live in the retrieval model's own input space (the leak-free
`[contrast, stellar-SNR]` observable, asinh-normed with the INARA **train** stats):

```
input  = encoded NOISY clouded observable   [B, 2, L]   (ch0 clouded contrast, ch1 SNR)
target = encoded NOISELESS clear contrast    [B, 1, L]   (the ideal restoration)
```

The SNR channel is **cloud-invariant** (its numerator `F_star = star_planet − planet`
is untouched by clouds), so it is passed through as per-λ reliability context. The net
predicts a **residual** on the contrast channel with a zero-initialised head, so it
*starts as the identity* and only has to learn the band depth the cloud muted.

Training uses the **grey** cloud family (randomised opacity `f`/brightening `b`) plus
exposure-time noise over α∈[0.3, 300]. The other families (`non_grey`,
`band_selective`, `patchy`) are **held out** so the eval measures generalisation to
unseen cloud physics rather than fitting it.

## Files

| file | role |
|---|---|
| `model.py` | `DecloudUNet1D` — residual 1D U-Net, `[B,2,L]→[B,1,L]`; `python -m cloud_recovery.model` shape self-test |
| `cloud_pairs.py` | paired `(cloudy→clear)` generator + held-out-family helper (read-only on `cache_v2`) |
| `train.py` | supervised Huber trainer → `src/cloud_recovery/checkpoints/` |
| `eval.py` | reconstruction + **frozen-retrieval R² recovery** test → `src/cloud_recovery/eval_out/` |

## Run order (in your terminal)

```bash
cd Exoplanets
export PYTHONPATH=src            # same convention as evaluation/ and common/

# 0. (instant) verify the network shape/identity-at-init — no data, no training
python -m cloud_recovery.model

# 1. sanity-check the training path on a subset (1 epoch)
python -m cloud_recovery.train --smoke

# 2. full training (resumable-free; writes decloud_seed0.pth)
python -m cloud_recovery.train --seed 0

# 3. the honest test: does de-clouding restore a FROZEN retrieval model's R²?
python -m cloud_recovery.eval --track transformerarch --seed 0 \
    --decloud-ckpt src/cloud_recovery/checkpoints/decloud_seed0.pth --plot-examples 4
```

The eval prints, per cloud family × exposure α, three numbers on the **same frozen
checkpoint**:

| input | expected R² |
|---|---|
| clear | baseline (high) |
| cloudy | drops sharply (cloud–abundance degeneracy) |
| **declouded** | recovers most of the gap ← the win |

`R²_recovery = (R²_declouded − R²_cloudy)/(R²_clear − R²_cloudy)`. If no retrieval
checkpoint is present the eval still reports reconstruction recovery and can plot
example spectra.

## Honest caveat

Heavy clouds (`f→1`) physically **erase** band depth; no network can recover erased
information, so the declouder will restore moderate clouds well and increasingly make a
*plausible-average guess* under heavy opacity. That is fine for a diagnostic /
exploration tool and is precisely why the output is inference-only. Read `recon_recovery`
and the held-out-family R² together — a big grey→held-out drop means the declouder
learned the grey operator, not "clouds" in general.

This is the **restoration front-end** answer to clouds. We also tried the alternative
**robust-training** answer (retraining retrieval models directly on cloud-augmented
data); it needed very long training and made accuracy worse, so it was retired
(kept for provenance under `junk/legacy_src/cloud_robust/`). See `report/REPORT.tex`
§6 for the comparison.
