"""Section C — Cross-generator domain gap (petitRADTRANS / TauREx / MultiREx).

Scores the INARA-trained weights on a test split from a DIFFERENT physics forward
model, through the identical v2 eval path (the engine cache symlinks the INARA-train
norm/labels, so nothing is refit). The headline is the covered-species mean R² at the
noiseless contrast reference (the domain-shift number, free of the α=1 photon floor).

Offline-safe: iterates the engine caches that EXIST under data/ and skips the rest.
Build them with (in the engine venv, then venvNLP):
  build_eval_cache.py --engine {prt,taurex,multirex} --feature-mode both  →  build_cache_v2_prt.py
"""
from __future__ import annotations

import os

from evaluation import core, plots

SECTION, SUITE = "C", "cross_generator"
TITLE = "Cross-generator (pRT / TauREx / MultiREx)"
EPISTEMIC = "Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics."

# (engine label, cache dir). Covered species = the 5 all these 0.2-2µm engines imprint.
ENGINE_CACHES = [
    ("prt", os.path.join(core.REPO, "data", "cache_v2_prt")),
    ("taurex", os.path.join(core.REPO, "data", "cache_v2_taurex")),
    ("multirex", os.path.join(core.REPO, "data", "cache_v2_multirex")),
]


def _present():
    return [(e, d) for e, d in ENGINE_CACHES if os.path.exists(os.path.join(d, "test_x.pt"))]


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    present = _present()
    if not present:
        return False, ("no cross-gen caches under data/cache_v2_{prt,taurex,multirex}; "
                       "build with build_eval_cache.py + build_cache_v2_prt.py")
    return True, ""


def _eval_engine(ctx, engine, cache_dir):
    ref = core.predict_cache(ctx, cache_dir, noiseless=True)
    scref = core.score(ref["y_true"], ref["ens"], bootstrap=0)
    out1 = core.predict_cache(ctx, cache_dir, alpha=1.0)
    sc1 = core.score(out1["y_true"], out1["ens"], bootstrap=1000)
    raw_x, noise = out1["raw_x"], out1["noise"]
    sweep = []
    for a in ctx.alphas:
        oa = core.predict_cache(ctx, cache_dir, alpha=a)
        sca = core.score(oa["y_true"], oa["ens"], bootstrap=0)
        sweep.append(dict(alpha=float(a), **core.snr_summary(raw_x[:, 0], raw_x[:, 1], noise, a),
                          r2_covered=sca["r2_covered"], r2_all12=sca["r2_all12"]))
    return {"engine": engine, "cache": cache_dir, "n": int(len(ref["y_true"])),
            "r2_covered_ref": scref["r2_covered"], "r2_all12_ref": scref["r2_all12"],
            "r2_covered_alpha1": sc1["r2_covered"], "per_species_alpha1": sc1["per_species"],
            "sweep": sweep}


def run(ctx):
    # native (INARA) noiseless covered ref for context on the gap
    nat = core.predict_cache(ctx, ctx.cache_v2, noiseless=True)
    native_ref = core.score(nat["y_true"], nat["ens"], bootstrap=0)["r2_covered"]

    results, tables, artifacts = [], [], []
    for engine, cache_dir in _present():
        r = _eval_engine(ctx, engine, cache_dir)
        results.append(r)
        try:
            artifacts.append(plots.sweep_plot(ctx.out_dir(SUITE), f"{ctx.label}-{engine}",
                                              r["sweep"], r["r2_covered_ref"],
                                              fname=f"r2_vs_snr_{engine}.png"))
        except Exception:
            pass
    tables.append({
        "name": "Covered-species R²(log10): native INARA vs each generator",
        "columns": ["generator", "N", "R²_covered noiseless-ref", "R²_covered @α=1",
                    "gap vs INARA (ref)"],
        "rows": [["INARA (native)", "-", native_ref, "-", 0.0]] +
                [[r["engine"], r["n"], r["r2_covered_ref"], r["r2_covered_alpha1"],
                  (r["r2_covered_ref"] - native_ref)] for r in results],
    })
    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={"native_R2_covered_ref": native_ref,
                  **{f"{r['engine']}_R2_covered_ref": r["r2_covered_ref"] for r in results}},
        tables=tables,
        caveats=[
            "A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale "
            "shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG "
            "anchor and decompose it with Section-J honesty stats before quoting it.",
        ],
        artifacts=artifacts, provenance=ctx.provenance(),
        data={"native_ref": native_ref, "engines": results},
    )
    return core.write_result(ctx, res)
