"""Section H — Comparison to published literature retrievals (PSEUDO-truth).

For each benchmark exoplanet (benchmarks/published_retrievals.json) a spectrum is
synthesised from its REAL published system parameters + literature-retrieved abundances
(via the in-env band-template engine), scored through the v2 pipeline, and the model's
prediction is compared to the published retrieval on the retrieved species.

Epistemic status is graded by ``domain_match``: transmission/emission hot Jupiters are
FAR (a different observable — agreement is not accuracy); directly-imaged young giants
(51 Eri b, HR 8799 e) are NEAR (reflected+thermal, the least out-of-domain). The report
separates the two groups and never treats a hot-Jupiter number as an accuracy claim.
"""
from __future__ import annotations

import os

import numpy as np

from evaluation import core, synth, known_targets
from evaluation import metrics_extra as mx
from common.data import TARGET_COLUMNS

import sys
sys.path.insert(0, os.path.join(core.SRC, "crossgen_eval"))
from inara_grid import load_inara_grid   # noqa: E402
from evaluation.engines import reflected_engine as eng   # noqa: E402

SECTION, SUITE = "H", "published_retrieval"
TITLE = "Published-retrieval comparison (benchmark exoplanets)"
EPISTEMIC = "Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy."


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(os.path.join(core.BENCH_DIR, "published_retrievals.json")):
        return False, "benchmarks/published_retrievals.json missing"
    return True, ""


def run(ctx):
    wl = load_inara_grid()
    targets = known_targets.published_targets()
    rows, data = [], {}
    for t in targets:
        ps = eng.generate_planet_shape(t["planet"], wl)
        ss = eng.stellar_continuum_shape(t["planet"], wl)
        raw_x, noise = synth.build_raw(ps[None], ss[None], ctx.cache_v2, seed=0)
        pred = core.predict_raw(ctx, raw_x, noise, noiseless=True)["ens"][0]

        # dex error on the RETRIEVED species only (the pseudo-truth is defined there)
        idx = [TARGET_COLUMNS.index(s) for s in t["retrieved_species"] if s in TARGET_COLUMNS]
        de = mx.dex_error(pred, t["true_vector"])
        mean_de = float(np.mean(de[idx])) if idx else None
        rho = mx.abundance_spearman(pred[idx], t["true_vector"][idx]) if len(idx) > 1 else None
        data[t["name"]] = {"observable": t["observable"], "domain_match": t["domain_match"],
                           "mean_dex_error_retrieved": mean_de, "ordering_rho": rho,
                           "retrieved_species": t["retrieved_species"], "doi": t["doi"],
                           "pred": {c: float(pred[i]) for i, c in enumerate(TARGET_COLUMNS)}}
        rows.append([t["name"], t["domain_match"], t["observable"],
                     ",".join(t["retrieved_species"]),
                     round(mean_de, 2) if mean_de is not None else "-",
                     round(rho, 2) if rho is not None else "-"])

    near = [d for d in data.values() if d["domain_match"] == "near"]
    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={"n_targets": len(targets),
                  "n_near-domain (direct-imaging)": len(near),
                  "mean dex-err (near-domain)":
                      float(np.mean([d["mean_dex_error_retrieved"] for d in near
                                     if d["mean_dex_error_retrieved"] is not None])) if near else None},
        tables=[{"name": "Model vs published retrieval (dex error on retrieved species)",
                 "columns": ["target", "domain", "observable", "retrieved", "mean dex-err", "ρ"],
                 "rows": rows}],
        caveats=[
            "Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with "
            "large error bars / degeneracies) — pseudo-truth, not ground truth.",
            "domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than "
            "this model's reflected-light domain: those rows are context, NOT accuracy. Only "
            "domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.",
            "Spectra are band-template proxies from published params; see reflected_engine docstring.",
        ],
        provenance=ctx.provenance(),
        data={"targets": data},
    )
    return core.write_result(ctx, res)
