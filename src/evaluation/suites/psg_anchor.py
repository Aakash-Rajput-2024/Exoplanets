"""Section D — PSG sanity anchor (the single most important cross-gen control).

Scores held-out REAL INARA/PSG spectra through the EXACT same eval path the
cross-generator caches use (median-match + real-noise bootstrap + INARA-fit
norm/labels). Only the generator is unchanged vs the pRT/TauREx caches — so if this
anchor's covered-species R² collapses, the eval PATH is the confound, not the
generator, and no cross-gen gap can be trusted until it's fixed.

Requires ``data/cache_v2_psg`` from ``crossgen_eval/build_cache_v2_psg.py`` (offline —
0 PSG API calls; it reuses the real held-out ``data/inara_1by3`` CSVs).

PASS: anchor covered-ref R² ≥ 0.9 × native INARA covered-ref R².
"""
from __future__ import annotations

import os

from evaluation import core

SECTION, SUITE = "D", "psg_anchor"
TITLE = "PSG sanity anchor (eval-path control)"
EPISTEMIC = "Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself."

PSG_CACHE = os.path.join(core.REPO, "data", "cache_v2_psg")


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(os.path.join(PSG_CACHE, "test_x.pt")):
        return False, ("data/cache_v2_psg missing; build with "
                       "`python src/crossgen_eval/build_cache_v2_psg.py` (offline)")
    return True, ""


def run(ctx):
    nat = core.predict_cache(ctx, ctx.cache_v2, noiseless=True)
    native_ref = core.score(nat["y_true"], nat["ens"], bootstrap=0)["r2_covered"]

    anc = core.predict_cache(ctx, PSG_CACHE, noiseless=True)
    anchor = core.score(anc["y_true"], anc["ens"], bootstrap=1000)
    anchor_ref = anchor["r2_covered"]

    ratio = (anchor_ref / native_ref) if native_ref not in (0, None) else None
    passed = (ratio is not None) and (ratio >= 0.9)

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label,
        status="ok" if passed else "preliminary", epistemic=EPISTEMIC,
        headline={"native_R2_covered_ref": native_ref, "anchor_R2_covered_ref": anchor_ref,
                  "anchor/native": ratio, "PASS(≥0.9×)": passed},
        tables=[{"name": "Per-species R²(log10) on the PSG anchor (α=1, 95% CI)",
                 "columns": ["species", "R²", "95% CI"],
                 "rows": [[c, anchor["per_species"][c]["r2"], anchor["per_species"][c].get("r2_ci")]
                          for c in core.TARGET_COLUMNS]}],
        caveats=[] if passed else [
            "⚠ ANCHOR BELOW 0.9× NATIVE: the eval path (median-match / noise bootstrap / symlink) "
            "is degrading the score, so any cross-generator gap in Section C is confounded and "
            "must NOT be attributed to generator physics until this passes.",
        ],
        provenance=ctx.provenance(),
        data={"native_ref": native_ref, "anchor": anchor, "ratio": ratio, "pass": passed},
    )
    return core.write_result(ctx, res)
