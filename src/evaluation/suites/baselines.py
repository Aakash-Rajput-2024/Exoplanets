"""Section B — Classical baselines (the reference floor for the neural R²).

Ingests the existing ``results_v2/baselines/summary.json`` (produced by
``common/baselines.py``: PriorMean / Ridge / RandomForest at the matched observable/α).
If absent it runs a quick SUBSAMPLED fit so the section still populates offline, and
compares the track's own α=1 covered/all-12 R² to the linear (Ridge) floor.

A neural R² that fails to beat Ridge means 'a linear model suffices' — the deep net is
unjustified. This section makes that comparison explicit in every report.
"""
from __future__ import annotations

import json
import os

from evaluation import core

SECTION, SUITE = "B", "baselines"
TITLE = "Classical baselines (PriorMean / Ridge / RandomForest)"
EPISTEMIC = "Ground truth: exact. The linear/prior information floor the neural net must beat."

_SUMMARY = os.path.join(core.REPO, "results_v2", "baselines", "summary.json")


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    return True, ""


def run(ctx, n_subsample=4000, skip_rf=False):
    base = _load_or_fit(ctx, n_subsample, skip_rf)
    # track's own α=1 covered/all-12 for a head-to-head row
    out = core.predict_cache(ctx, ctx.cache_v2, alpha=1.0)
    sc = core.score(out["y_true"], out["ens"], bootstrap=0)

    rows = [["PriorMean", base.get("PriorMean"), "no-information reference (R²≈0 line)"],
            ["Ridge", base.get("Ridge"), "linear information floor"],
            ["RandomForest", base.get("RandomForest"), "capacity-limited nonlinear ref"],
            [f"THIS: {ctx.label} (α=1)", sc["r2_all12"], "neural, same observable/α"]]
    ridge = base.get("Ridge")
    beats = (ridge is not None) and (sc["r2_all12"] > ridge)

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={"neural_all12_R2@a1": sc["r2_all12"], "Ridge_floor_R2": ridge,
                  "beats_linear_floor": beats},
        tables=[{"name": "Overall R²(log10), all-12, same observable & α",
                 "columns": ["model", "R²(log10)", "note"], "rows": rows}],
        caveats=[
            "Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the "
            "neural row is this track at α=1. For the rigorous paired-bootstrap-significant "
            "comparison at α*, see VALIDATION_PLAN A3.",
        ] + ([] if beats else
             ["⚠ This track's α=1 R² does NOT exceed the Ridge floor — at the photon floor "
              "that is expected; compare at the noiseless ref / α* before concluding."]),
        provenance=ctx.provenance(),
        data={"baselines": base, "neural_alpha1": sc},
    )
    return core.write_result(ctx, res)


def _load_or_fit(ctx, n_subsample, skip_rf):
    """Return {model: overall_r2_log10}. Prefer the full cached summary; else quick fit."""
    if os.path.exists(_SUMMARY):
        with open(_SUMMARY) as f:
            s = json.load(f)
        return {k: v.get("overall_r2_log10") for k, v in s.get("baselines", {}).items()}
    try:
        from common import baselines as B
        res = B.run_baselines(cache_dir=ctx.cache_v2, n_subsample=n_subsample, skip_rf=skip_rf)
        import numpy as np
        return {k: float(np.nanmean(v["r2"])) for k, v in res.items()}
    except Exception as e:
        return {"_error": f"baselines unavailable ({e}); run `python src/common/baselines.py`"}
