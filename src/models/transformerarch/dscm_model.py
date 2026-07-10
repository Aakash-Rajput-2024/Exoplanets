"""LEGACY STUB — the VAE-DSCM classes formerly defined here have been removed.

This module used to define DSCMEncoder/DSCMDecoder/DSCM: a conditional VAE that
learned S = g(Z, A, E) + noise to synthesize "same atmosphere, different
environment" counterfactual spectra for the causal track. It is dead code in
the v2 pipeline:
  * Nothing in the active pipeline imports it. registry.py loads only
    NasaInaraTransformer (by file path) for every track, including "causal".
  * It was superseded by exact environment re-pairing, which needs no learned
    decoder at all (see src/common/counterfactuals.py for the full derivation
    and rationale for why the VAE was dropped).
  * A byte-for-byte DIVERGED copy of these same classes also existed inside
    src/models/causal/cnn_trnas/model.py -- identical class names / state_dict shapes,
    but different hardcoded decoder upsample sizes (512/2048 here vs.
    sequence-derived 547/2189 there). Two modules with the same class name
    silently computing different forward() outputs from the same checkpoint is
    exactly the kind of bug this cleanup removes. That copy has been deleted
    too (src/models/causal/cnn_trnas/model.py now defines only PositionalEncoding and
    NasaInaraTransformer).

Only src/models/transformerarch/train_dscm.py (itself legacy/unused by the v2
pipeline) imported ``DSCM`` from this file; it is now dead code as a
consequence and is not being run. The file is kept (rather than deleted) only
so that stale import path resolves to an explicit, documented dead end instead
of a bare ModuleNotFoundError, and so `git log`/`git blame` on this path keeps
its history.
"""
