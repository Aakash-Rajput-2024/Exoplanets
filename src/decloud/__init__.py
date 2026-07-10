"""Standalone spectral de-clouding front-end (isolated experiment).

A supervised 1D restoration network that maps a CLOUDED reflected-light contrast
spectrum back to an estimate of its CLEAR counterpart, so a *frozen* retrieval
model can be run on the restored spectrum without ever being retrained on clouds.

WHY SUPERVISED (not a VAE)
    The cloud transform ``common.cloud_families`` is a KNOWN forward operator, so
    every clear INARA spectrum + a random cloud draw is an EXACT (cloudy, clear)
    training pair. With paired data a supervised U-Net directly minimises
    reconstruction error against the true clear spectrum — no latent-space
    ambiguity, no risk of the decoder inventing structure the way an unpaired VAE
    would. The declouder's output is used for INFERENCE ONLY; it is never fed back
    into retrieval *training* (that would inject the net's prior as if it were
    data — the exact contamination this module is designed to avoid).

ISOLATION GUARANTEE
    This package only ever READS the shared pipeline (``common.*``) and the
    ``data/cache_v2`` cache. It writes solely under ``src/decloud/`` (checkpoints,
    eval outputs, a small continuum cache). It imports no training entry point and
    mutates no existing module, cache, or checkpoint. Deleting this folder leaves
    the rest of the repo byte-for-byte unchanged.

LAYOUT
    cloud_pairs.py  paired (cloudy→clear) encoded-observable generator
    model.py        DecloudUNet1D — [B,2,L] cloudy obs -> [B,1,L] clear contrast
    train.py        supervised trainer      (PYTHONPATH=src python -m decloud.train ...)
    eval.py         reconstruction + FROZEN-retrieval R² recovery test
"""
