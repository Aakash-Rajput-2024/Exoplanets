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
    lambda_inv=1.0,                # strength of the counterfactual-invariance penalty
                                   # (only used when --cf-invariance is set; see the
                                   # causal_cfi track and common/counterfactuals.py).
    seeds=(0, 1, 2),               # H6
)


# --- Track registry: architecture only ---------------------------------------
TRACKS = {
    "original1dcnn": dict(
        model_py=f"{SRC}/models/original1dcnn/model.py", model_cls="NasaInaraModel",
        desc="Paper-baseline 4-conv CNN (36M-param FC head)",
    ),
    # NOTE: '2channel1dcnn' was REMOVED (2026-07-08 audit). Under the shared
    # matched-budget harness it is a byte-for-byte duplicate of original1dcnn
    # (in_channels forced to 2 for every track; identical model file; bit-identical
    # predictions to 17 digits) — it was one baseline double-counted as two. Its
    # historical role (1- vs 2-channel input) no longer exists in v2. The stale
    # src/2channel1dcnn/ directory is kept for provenance but must NOT be reported
    # as an independent architecture.
    "optimized1dcnn": dict(
        model_py=f"{SRC}/models/optimized1dcnn/model.py", model_cls="NasaInaraModel",
        desc="CNN + BatchNorm + progressive kernels + AdaptiveAvgPool",
    ),
    "transformerarch": dict(
        model_py=f"{SRC}/models/transformerarch/model.py", model_cls="NasaInaraTransformer",
        desc="CNN downsample ×8 → 2-layer Transformer → GAP → MLP",
    ),
    # Causal ARCHITECTURE == transformer (identical backbone). The three causal
    # variants differ ONLY in the training objective, so the comparison is a clean
    # ablation of the causal machinery, not of capacity:
    #   causal      : --cf              exact do(env) counterfactual AUGMENTATION
    #                 (re-paired host star + noise; the old, weaker "just show more
    #                  environments" behaviour — kept as the augmentation baseline).
    #   causal_cfi  : --cf-invariance   exact do(env) counterfactual INVARIANCE
    #                 objective — penalises the retrieval prediction for CHANGING
    #                 under the intervention: L = task + λ·‖f(x) − f(x^do(env=j))‖².
    #                 Because INARA draws atmosphere ⟂ environment, the intervention
    #                 is EXACT (not a learned/approximate environment as in IRM/DG),
    #                 so the invariance target is correct by construction — this is
    #                 the track's methodological novelty. See common/counterfactuals.py.
    # Train the plain-transformer control with neither flag for the ablation.
    "causal": dict(
        model_py=f"{SRC}/models/causal/cnn_trnas/model.py", model_cls="NasaInaraTransformer",
        desc="Transformer backbone; exact do(env) counterfactual augmentation (--cf)",
    ),
    "causal_cfi": dict(
        model_py=f"{SRC}/models/causal/cnn_trnas/model.py", model_cls="NasaInaraTransformer",
        desc="Transformer backbone; exact-intervention counterfactual-invariance objective (--cf-invariance)",
    ),
    # ~2M-param dedicated do-calculus backbone (wider/deeper transformer + attention
    # pooling). Intended headline model for the invariance objective; train with:
    #   python -m common.train_runner causal_xl --cf-invariance --seed 0
    "causal_xl": dict(
        model_py=f"{SRC}/models/causal/cnn_trnas/causal_model.py", model_cls="NasaInaraCausalNet",
        desc="~2M-param transformer for the do-calculus counterfactual-invariance objective (--cf-invariance)",
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
