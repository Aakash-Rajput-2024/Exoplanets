"""MultiREx generator adapter for cross-generator evaluation.

MultiREx (local: ``MultiREx-public/``) wraps TauREx 3 with an instrument/SNR simulator.
Two things it provides here, both requiring ``MultiREx-public/.venv`` (TauREx opacities),
so this module is imported by the OFFLINE GENERATION step, not the torch eval env:

  * ``generate_planet_shape`` / ``stellar_continuum_shape`` — the reflected-light proxy
    (identical construction to ``taurex_engine`` since that is already MultiREx-backed);
    exposed under the ``multirex`` engine name so ``build_eval_cache.py --engine multirex``
    yields a cross-gen cache scored in Section C alongside pRT/TauREx.

  * ``native_transmission_shape`` — MultiREx's NATIVE observable: a transit-depth
    (transmission) spectrum. This is the WRONG observable for the reflected-light model, so
    it feeds the Section-G OOD probe (or a labelled OOD generator), never the accuracy path.

Keeping the reflected proxy identical to taurex_engine (rather than a second copy) makes
the 'MultiREx vs TauREx' cross-gen an honest instrument/sampling comparison on the same
radiative transfer, not an accidental physics difference.
"""
from __future__ import annotations

import numpy as np

# The reflected-light proxy is defined once, in taurex_engine (MultiREx/TauREx-backed).
from evaluation.engines.taurex_engine import (   # noqa: F401
    generate_planet_shape, stellar_continuum_shape, _transmission_depth,
)


def native_transmission_shape(p, wl_grid_um):
    """MultiREx native transit-depth (Rp/Rs)² spectrum on wl_grid_um.

    A transiting-geometry observable — provided for the OOD probe (Section G), NOT for
    reflected-light retrieval accuracy. Returned as 1 − normalised depth so it reads as a
    'brightness' the encoder can ingest while remaining clearly off-distribution.
    """
    depth = _transmission_depth(p, wl_grid_um)
    d = np.asarray(depth, dtype=float)
    rng = d.max() - d.min()
    return np.nan_to_num(1.0 - (d - d.min()) / (rng + 1e-30))
