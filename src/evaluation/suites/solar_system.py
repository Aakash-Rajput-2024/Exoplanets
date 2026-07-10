"""Section E — Solar-System bodies observed as exoplanets (LITERAL ground truth).

The only real-target evaluation where composition is exactly known. Earth/Mars/Venus/
Titan are terrestrial and REPRESENTABLE in the 12-species simplex; the H2/He giants
(Jupiter/Saturn/Uranus/Neptune) are NOT representable (their mass sits in H2/He, not a
target) and are used as an HONESTY probe — the model must not fabricate high O2/O3.

Each body's spectrum is synthesised by the in-env band-template reflected engine
(evaluation/engines/reflected_engine.py — a transparent PROXY; see its docstring),
median-matched to the INARA scale, and scored at the noiseless reference (best-case
information) and at the α=1 photon floor. Too few targets for R² → per-target dex error,
'did it get the dominant gas', and abundance-ordering Spearman.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from evaluation import core, synth, known_targets, plots
from evaluation.engines import reflected_engine as eng
from evaluation import metrics_extra as mx
from common.data import TARGET_COLUMNS

sys.path.insert(0, os.path.join(core.SRC, "crossgen_eval"))
from inara_grid import load_inara_grid   # noqa: E402

SECTION, SUITE = "E", "solar_system"
TITLE = "Solar-System-as-exoplanet (known composition)"
EPISTEMIC = "Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test."

O2O3_IDX = [TARGET_COLUMNS.index("O2"), TARGET_COLUMNS.index("O3")]


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(os.path.join(ctx.cache_v2, "test_x.pt")):
        return False, "INARA cache_v2 (for scale/noise/norm) missing"
    return True, ""


def _predict_target(ctx, planet, wl, seed=0):
    ps = eng.generate_planet_shape(planet, wl)
    ss = eng.stellar_continuum_shape(planet, wl)
    raw_x, noise = synth.build_raw(ps[None], ss[None], ctx.cache_v2, seed=seed)
    ref = core.predict_raw(ctx, raw_x, noise, noiseless=True)
    a1 = core.predict_raw(ctx, raw_x, noise, alpha=1.0)
    return ref["ens"][0], a1["ens"][0]


def run(ctx):
    wl = load_inara_grid()
    targets = known_targets.solar_system_targets()

    rows, honesty, per_target, artifacts = [], [], {}, []
    for t in targets:
        pred_ref, pred_a1 = _predict_target(ctx, t["planet"], wl)
        summ = mx.known_truth_summary(pred_ref, t["true_vector"], TARGET_COLUMNS, t["covered_species"])
        o2o3 = float(pred_ref[O2O3_IDX].sum())
        per_target[t["name"]] = {"representable": t["representable"], "summary": summ,
                                 "o2_plus_o3_pred": o2o3, "note": t["note"]}
        tag = "" if t["representable"] else " (giant/honesty)"
        rows.append([t["name"] + tag, summ["dominant_true"], summ["dominant_pred"],
                     "✓" if summ["dominant_correct"] else "✗",
                     round(summ["mean_dex_error_covered"], 2)
                     if summ["mean_dex_error_covered"] is not None else "-",
                     round(summ["spearman_ordering"], 2)])
        if not t["representable"]:
            honesty.append([t["name"], round(o2o3, 4),
                            "OK (low)" if o2o3 < 0.1 else "⚠ fabricated O2/O3"])
        try:
            artifacts.append(plots.truth_bars(ctx.out_dir(SUITE), t["name"], TARGET_COLUMNS,
                                             t["true_vector"], pred_ref, t["covered_species"],
                                             fname=f"truth_{t['name']}.png"))
        except Exception:
            pass

    repr_targets = [t for t in targets if t["representable"]]
    n_dom = sum(per_target[t["name"]]["summary"]["dominant_correct"] for t in repr_targets)
    tables = [{"name": "Per-target recovery (noiseless reference)",
               "columns": ["target", "dominant true", "dominant pred", "dom✓",
                           "mean dex-err (covered)", "ordering ρ"], "rows": rows}]
    if honesty:
        tables.append({"name": "Honesty probe — giants must NOT show high O2/O3",
                       "columns": ["giant", "pred O2+O3", "verdict"], "rows": honesty})

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={"dominant-gas correct (terrestrial)": f"{n_dom}/{len(repr_targets)}",
                  "mean dex-err covered (terrestrial)":
                      float(np.mean([per_target[t["name"]]["summary"]["mean_dex_error_covered"]
                                     for t in repr_targets
                                     if per_target[t["name"]]["summary"]["mean_dex_error_covered"] is not None]))},
        tables=tables,
        caveats=[
            "Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this "
            "tests whether the model responds to which gases are present, not absolute fidelity. "
            "For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.",
            "Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, "
            "not an accuracy test.",
            "INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like "
            "ppb trace gases are out-of-distribution; read the MAJOR/covered gases.",
        ],
        artifacts=artifacts, provenance=ctx.provenance(),
        data={"per_target": per_target},
    )
    return core.write_result(ctx, res)
