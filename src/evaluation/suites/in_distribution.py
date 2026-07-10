"""Section A — In-distribution test-set evaluation.

INARA/PSG held-out TEST split (the frozen split the model never selected on): overall
& per-species R²(log10) at the α=1 photon floor, the noiseless contrast reference, and
the R²-vs-exposure (SNR) sweep — the C1 headline. Computed through the shared v2 core so
the numbers match every other section.

The canonical rich standalone artifacts (12-panel scatter, styled sweep PNG) come from
``python -m common.evaluate <track>``; this suite produces the compact report row + JSON
and a sweep plot, and is suffix-aware (e.g. a `_grey` variant) which evaluate_track is not.
"""
from __future__ import annotations

import os

import numpy as np

from evaluation import core, plots
from common.evaluate import TARGET_COLUMNS, MAX_TRAINED_ALPHA

SECTION, SUITE = "A", "in_distribution"
TITLE = "In-distribution (INARA held-out test)"
EPISTEMIC = "Ground truth: exact (synthetic). The information ceiling of the observable."


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(os.path.join(ctx.cache_v2, "test_x.pt")):
        return False, f"INARA test cache missing ({ctx.cache_v2})"
    return True, ""


def run(ctx):
    out1 = core.predict_cache(ctx, ctx.cache_v2, alpha=1.0)
    sc1 = core.score(out1["y_true"], out1["ens"], bootstrap=1000)
    ref = core.predict_cache(ctx, ctx.cache_v2, noiseless=True)
    scref = core.score(ref["y_true"], ref["ens"], bootstrap=0)

    raw_x, noise = out1["raw_x"], out1["noise"]
    sweep = []
    for a in ctx.alphas:
        oa = core.predict_cache(ctx, ctx.cache_v2, alpha=a)
        sca = core.score(oa["y_true"], oa["ens"], bootstrap=0)
        row = dict(alpha=float(a), **core.snr_summary(raw_x[:, 0], raw_x[:, 1], noise, a))
        row["r2_covered"] = sca["r2_covered"]
        row["r2_all12"] = sca["r2_all12"]
        sweep.append(row)

    # per-seed spread (reproducibility) when >1 seed is available
    per_seed = [float(np.nanmean(core.metrics.report(out1["y_true"], p, space="log10",
                                                     bootstrap=0)["r2"])) for p in out1["preds"]]

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label,
        status="ok" if ctx.n_seeds >= 3 else "preliminary",
        epistemic=EPISTEMIC,
        headline={
            "R2_covered @a=1": sc1["r2_covered"],
            "R2_all12 @a=1": sc1["r2_all12"],
            f"R2_covered noiseless-ref (a={MAX_TRAINED_ALPHA:g})": scref["r2_covered"],
            f"R2_all12 noiseless-ref": scref["r2_all12"],
            "RMSE_all12 (dex) @a=1": sc1["rmse_all12_dex"],
            "n_seeds": ctx.n_seeds,
            "per_seed_R2_all12_mean±std": [float(np.mean(per_seed)),
                                           float(np.std(per_seed)) if ctx.n_seeds > 1 else None],
        },
        tables=[
            {"name": "Per-species R²(log10) @ α=1 (95% bootstrap CI)",
             "columns": ["species", "R²", "95% CI", "RMSE(dex)", "MAE(dex)"],
             "rows": [[c, sc1["per_species"][c]["r2"], sc1["per_species"][c].get("r2_ci"),
                       sc1["per_species"][c]["rmse_dex"], sc1["per_species"][c]["mae_dex"]]
                      for c in TARGET_COLUMNS]},
            {"name": "SNR sweep (α = √(t/t_nom))",
             "columns": ["alpha", "exposure×", "SNR_planet(band)", "R²_covered", "R²_all12"],
             "rows": [[r["alpha"], r["exposure_x"], r["snr_planet_band"],
                       r["r2_covered"], r["r2_all12"]] for r in sweep]},
        ],
        caveats=[
            "α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); "
            "read the noiseless reference and the sweep, not the single α=1 number.",
        ] + (["PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only."]
             if ctx.n_seeds < 3 else []),
        provenance=ctx.provenance(),
        data={"alpha1": sc1, "noiseless_ref": scref, "sweep": sweep, "per_seed_r2": per_seed},
    )
    try:
        p = plots.sweep_plot(ctx.out_dir(SUITE), ctx.label, sweep, scref["r2_covered"])
        res.artifacts.append(p)
    except Exception as e:                       # plotting must never break a suite
        res.caveats.append(f"(plot skipped: {e})")
    return core.write_result(ctx, res)
