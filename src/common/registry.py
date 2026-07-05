"""Track registry + the single matched training budget (fixes C4, M5, M1, M3).

C4's core defect: original/2channel/optimized ran 4 epochs of L1 at batch 1024;
the transformer ran 50 epochs of MSE at batch 16 with warmup+cosine; causal ran
on 13% more data. "Transformer beats CNN" and "causal beats transformer" were
therefore not attributable to architecture. The fix is structural, not
procedural: every track pulls the SAME budget from ``MATCHED`` and differs ONLY
in its model class. To vary the budget you must edit it in one place, for all
tracks at once — the confound cannot silently reappear.

Choices, with reasons:
  * loss = Huber (M1): unifies the L1/MSE split; in standardized CLR space the
    targets are O(1) so δ=1 behaves like MSE near zero, L1 in the tails (robust
    to the occasional noise-dominated blue-end spectrum).
  * batch = 64: equalizes the effective-LR/gradient-noise regime across
    architectures at a size that FITS. The transformer's attention memory scales
    as batch×seq² (seq=547), so batch 256 OOMs a 22 GB MPS (~19 GB) and batch 128
    hits a thrashing cliff (~7 s/batch); 64 uses ~7.4 GB with headroom and is
    still 4× the transformer's original batch-16. The CNNs are memory-light and
    unaffected. (Per-epoch wall-time is ~batch-independent below the cliff, since
    total work is fixed — so the transformer stays the ~10 min/epoch long pole.)
  * weight_decay = 1e-4 (M3): the CNNs had none.
  * cosine warmup + grad-clip for all: previously transformer-only.
  * seeds = (0,1,2) (H6): three runs per track → mean±std + paired bootstrap.

Model classes are loaded by FILE PATH (three tracks each define a class named
``NasaInaraModel``); importlib-by-path avoids the module-name collision, exactly
as ``cloud_robust/train_common.py`` already does.
"""

from __future__ import annotations

import importlib.util
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO = os.path.abspath(os.path.join(SRC, os.pardir))
CACHE_V2 = os.path.join(REPO, "data", "cache_v2")


# --- The one budget every track shares (C4) ---------------------------------
MATCHED = dict(
    obs_mode="contrast_snr",       # C1: physical observable (contrast + per-λ SNR)
    label_transform="clr",         # C2: simplex-correct labels
    input_norm="per_lambda_asinh",  # H3: per-λ asinh encoding
    alpha=1.0,                     # VALIDATED nominal: α=√(t/t_nom), t_nom = LUVOIR-like
                                   # 8 hr @ 5 pc ∝d² (Zorzan+25 §3.1). α=1 = literal
                                   # photon floor; headline = the eval α-sweep, not α=1.
    alpha_train_range=(0.3, 300.0),  # exposure-time augmentation: train/val draw α per
                                   # sample log-uniform over the eval-sweep range so ONE
                                   # model spans all exposures (else the sweep humps then
                                   # DECLINES at high SNR — a train/eval covariate-shift
                                   # artifact, not physics). See notes Part 3 / journal C10.
    loss="huber",                  # M1
    epochs=50,
    batch_size=64,                 # largest that fits the transformer on 22 GB MPS
    lr=1e-3,
    weight_decay=1e-4,             # M3
    warmup_epochs=5,
    grad_clip=1.0,
    patience=15,
    seeds=(0, 1, 2),               # H6
)


# --- Track registry: architecture only ---------------------------------------
TRACKS = {
    "original1dcnn": dict(
        model_py=f"{SRC}/original1dcnn/model.py", model_cls="NasaInaraModel",
        desc="Paper-baseline 4-conv CNN (36M-param FC head)",
    ),
    "2channel1dcnn": dict(
        model_py=f"{SRC}/2channel1dcnn/model.py", model_cls="NasaInaraModel",
        desc="Same net as original (kept for continuity; identical arch)",
    ),
    "optimized1dcnn": dict(
        model_py=f"{SRC}/optimized1dcnn/model.py", model_cls="NasaInaraModel",
        desc="CNN + BatchNorm + progressive kernels + AdaptiveAvgPool",
    ),
    "transformerarch": dict(
        model_py=f"{SRC}/transformerarch/model.py", model_cls="NasaInaraTransformer",
        desc="CNN downsample ×8 → 2-layer Transformer → GAP → MLP",
    ),
    # Causal ARCHITECTURE == transformer. Trained here WITHOUT counterfactuals it
    # is control (a) from C4 ("baseline transformer, identical schedule as
    # causal"). The counterfactual-augmented and placebo variants require the
    # DSCM to be regenerated in the new observable space (deferred; see journal).
    "causal": dict(
        model_py=f"{SRC}/causal/cnn_trnas/model.py", model_cls="NasaInaraTransformer",
        desc="Transformer backbone; DSCM counterfactual augmentation (base only here)",
    ),
}


def track_config(track: str) -> dict:
    """Full resolved config for a track = MATCHED budget + its architecture entry."""
    if track not in TRACKS:
        raise SystemExit(f"Unknown track '{track}'. Choose from {list(TRACKS)}.")
    cfg = dict(MATCHED)
    cfg.update(TRACKS[track])
    cfg["track"] = track
    return cfg


def load_model_class(model_py: str, cls_name: str):
    """Import a uniquely-named module from a file path and return the class."""
    mod_name = "reg_" + os.path.basename(os.path.dirname(model_py)) + "_model"
    spec = importlib.util.spec_from_file_location(mod_name, model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)
