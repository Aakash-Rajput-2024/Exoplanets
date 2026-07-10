"""Section J — OOD honesty statistics (VALIDATION_PLAN B6).

Decomposes a cross-generator score into (a) a per-λ standardised input shift δ(λ) and
variance ratio v(λ) vs INARA after the INARA-fit norm, and (b) per-species
R²_raw | R²_debiased | r²_pearson. This turns a single ambiguous negative cross-gen R²
into an honest diagnosis:

  * mean r²_pearson ≤ 0.10 and R²_debiased ≤ 0  → genuinely no transfer (failure = signal).
  * mean r²_pearson ≥ 0.30 (or R²_debiased ≫ R²_raw) → BIASED extrapolation (fixable by a
    domain-robust normalisation), which a raw single-number R² would have mislabeled.

Runs on whichever cross-gen / anchor caches are present (offline-safe).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from evaluation import core, plots
from evaluation import metrics_extra as mx

sys.path.insert(0, os.path.join(core.SRC, "evaluation/crossgen"))
from inara_grid import load_inara_grid   # noqa: E402

SECTION, SUITE = "J", "ood_honesty"
TITLE = "OOD honesty (δ/v, raw vs debiased R²)"
EPISTEMIC = "Decomposes cross-gen R² into input shift + bias vs genuine information loss."

CACHES = [
    ("prt", os.path.join(core.REPO, "data", "cache_v2_prt")),
    ("taurex", os.path.join(core.REPO, "data", "cache_v2_taurex")),
    ("multirex", os.path.join(core.REPO, "data", "cache_v2_multirex")),
    ("psg", os.path.join(core.REPO, "data", "cache_v2_psg")),
]


def _present():
    return [(e, d) for e, d in CACHES if os.path.exists(os.path.join(d, "test_x.pt"))]


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not _present():
        return False, "no cross-gen/anchor caches present (data/cache_v2_{prt,taurex,multirex,psg})"
    return True, ""


def run(ctx):
    wl = load_inara_grid()[:-1]     # trimmed grid matches encoded length
    enc_inara, _, _, _, _ = core.encode_cache(ctx, ctx.cache_v2, noiseless=True)

    rows, data, artifacts = [], {}, []
    for engine, cache_dir in _present():
        enc_gen, _, _, _, _ = core.encode_cache(ctx, cache_dir, noiseless=True)
        shift = mx.standardized_shift(enc_gen, enc_inara, channel=0)
        out = core.predict_cache(ctx, cache_dir, noiseless=True)
        db = mx.debiased_r2(out["y_true"], out["ens"], core.TARGET_COLUMNS, core.COVERED_DEFAULT)

        # B6 precedence: genuine-no-transfer is decided first (pearson AND debiased both fail),
        # so a large bias-removal gain from a very-negative raw R² can't mislabel a run whose
        # ordering correlation is ~0 as 'fixable'.
        if db["mean_r2_pearson"] <= 0.10 and db["mean_r2_debiased"] <= 0:
            verdict = "no transfer (genuine)"
        elif db["mean_r2_pearson"] >= 0.30 or db["mean_r2_debiased"] > 0:
            verdict = "biased extrapolation (fixable)"
        else:
            verdict = "partial"
        data[engine] = {"shift": {k: v for k, v in shift.items() if not k.endswith("lambda")},
                        "debiased_r2": {k: v for k, v in db.items() if k != "per_species"},
                        "verdict": verdict}
        rows.append([engine, round(shift["median_abs_delta"], 2), round(shift["median_v_ratio"], 2),
                     round(shift["frac_lambda_in_family"], 2), round(db["mean_r2_raw"], 3),
                     round(db["mean_r2_debiased"], 3), round(db["mean_r2_pearson"], 3), verdict])
        try:
            artifacts.append(plots.delta_lambda_plot(ctx.out_dir(SUITE), f"{ctx.label}-{engine}",
                                                    wl, shift["delta_lambda"], shift["v_lambda"],
                                                    fname=f"ood_delta_{engine}.png"))
        except Exception:
            pass

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={f"{e}_verdict": data[e]["verdict"] for e, _ in _present()},
        tables=[{"name": "OOD decomposition (covered species)",
                 "columns": ["engine", "med|δ|", "med v", "frac λ in-family",
                             "R²_raw", "R²_debiased", "r²_pearson", "verdict"],
                 "rows": rows}],
        caveats=[
            "δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input "
            "sits inside INARA's distribution at that wavelength.",
            "A raw negative R² is never published without its debiased companion (this section).",
        ],
        artifacts=artifacts, provenance=ctx.provenance(),
        data={"engines": data},
    )
    return core.write_result(ctx, res)
