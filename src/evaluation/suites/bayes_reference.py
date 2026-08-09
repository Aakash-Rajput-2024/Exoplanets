"""Section K — Bayesian reference retrieval (the information ceiling).

Closes AUDIT_REPORT.md finding #4 ("no classical/Bayesian baseline … the R² is
unattributable"). Section B already gives the *floor* (PriorMean / Ridge / RF); this
gives the *ceiling*, and between them a neural R² finally means something.

THE REFERENCE
    T0 (``evaluation.bayes.prior_is``) is self-normalized importance sampling over the
    88k INARA training spectra. Those spectra are real PSG outputs drawn from INARA's
    own prior, so the result is the exact posterior — correct prior, exact forward
    model, no surrogate — and its posterior mean in log10 space is the Bayes-optimal
    point estimate for the R²(log10) the whole pipeline is scored on.

    Therefore, per species and per α:
        neural R² ≈ Bayesian R²  ⇒ INFORMATION-limited. The data does not carry the
                                   answer; this is not a model failure and must be
                                   excluded from "neural weakness" claims
                                   (VALIDATION_PLAN.md:50).
        neural R² <  Bayesian R²  ⇒ CAPACITY-limited. The information is there and the
                                   net is not extracting it.

    Because the posterior mean is the conditional mean, the two decompose exactly:
        MSE_neural = MSE_Bayes + E‖θ̂_neural − θ̂_Bayes‖²
    (the cross term vanishes by orthogonality, in log10 space). The second term is a
    direct measurement of the capacity gap and is reported as ``excess_mse``.

THE ESS GATE — enforced, not advised
    Prior-proposal importance sampling degenerates to 1-nearest-neighbour once the
    likelihood concentrates. At α=10 it reports a healthy-looking R² that is in fact an
    ESS≈1 readout of the library's density, and publishing it would be an error. Any α
    whose median ESS is below ``likelihood.ESS_MIN`` is reported as NOT a ceiling and
    the section is stamped ``preliminary``. The high-α ceiling needs T2.
"""
from __future__ import annotations

import json
import os

import numpy as np

from evaluation import core
from evaluation.bayes import likelihood as L
from common.data import TARGET_COLUMNS

SECTION, SUITE = "K", "bayes_reference"
TITLE = "Bayesian reference retrieval (information ceiling)"
EPISTEMIC = ("Ground truth: exact. Exact posterior under INARA's own prior and PSG's "
             "own forward model — the ceiling no estimator on these spectra can beat.")

RESULTS = os.path.join(core.REPO, "results_v2", "bayes_reference")
SUMMARY = os.path.join(RESULTS, "summary.json")
COVERED = ["H2O", "CO2", "O2", "CH4", "O3"]
# Below this the neural and Bayesian R² are called equal and the species is declared
# information-limited. Comfortably inside the bootstrap CIs core.score reports.
TIE_DEX = 0.02


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(SUMMARY):
        return False, ("no T0 reference — run "
                       "`PYTHONPATH=src python -m evaluation.bayes.run_prior_is`")
    return True, ""


def _load_reference():
    with open(SUMMARY) as f:
        doc = json.load(f)
    per_alpha = {}
    for a in doc["by_alpha"]:
        path = os.path.join(RESULTS, f"prior_is_a{float(a):g}.npz")
        if os.path.exists(path):
            per_alpha[float(a)] = np.load(path)
    return doc, per_alpha


def run(ctx, alphas=None):
    doc, npz = _load_reference()
    by_alpha = {float(k): v for k, v in doc["by_alpha"].items()}
    want = sorted(a for a in by_alpha if (alphas is None or a in set(alphas)))

    rows, verdict_rows = [], []
    payload, any_resolved = {}, False

    for a in want:
        ref = by_alpha[a]
        z = npz.get(a)
        if z is None:
            continue
        idx = z["idx"]
        # Same planets, same α, same noise realization: predict_cache injects noise on
        # the FULL test tensor at TEST_NOISE_SEED, exactly as prior_is does, so row i
        # here is row i there. Subsample only AFTER predicting.
        out = core.predict_cache(ctx, ctx.cache_v2, alpha=a)
        nn_log = np.log10(np.clip(out["ens"][idx], 1e-12, None))
        truth_log = z["truth_log"]
        bayes_log = z["post_mean_log"]

        nn_r2 = _r2(truth_log, nn_log)
        bay_r2 = _r2(truth_log, bayes_log)
        resolved = bool(ref["resolved"])
        any_resolved |= resolved

        # MSE_neural = MSE_Bayes + E||neural - Bayes||^2  (orthogonality, log10 space)
        excess = float(np.mean((nn_log - bayes_log) ** 2))
        mse_nn = float(np.mean((nn_log - truth_log) ** 2))
        mse_b = float(np.mean((bayes_log - truth_log) ** 2))

        ci = [TARGET_COLUMNS.index(s) for s in COVERED]
        rows.append([
            f"{a:g}", round(float(ref["ess_median"]), 1),
            "yes" if resolved else "NO (1-NN)",
            round(float(np.mean(nn_r2[ci])), 3), round(float(np.mean(bay_r2[ci])), 3),
            round(float(np.mean(nn_r2)), 3), round(float(np.mean(bay_r2)), 3),
            round(excess, 4),
        ])
        payload[f"alpha_{a:g}"] = {
            "resolved": resolved, "ess_median": ref["ess_median"],
            "ess_frac_ok": ref["ess_frac_ok"], "n_gated": ref.get("n_gated"),
            "neural_r2_covered": float(np.mean(nn_r2[ci])),
            "bayes_r2_covered": float(np.mean(bay_r2[ci])),
            "neural_r2_all12": float(np.mean(nn_r2)),
            "bayes_r2_all12": float(np.mean(bay_r2)),
            "excess_mse_dex2": excess, "mse_neural_dex2": mse_nn, "mse_bayes_dex2": mse_b,
            "fraction_of_ceiling": (float(np.mean(nn_r2) / np.mean(bay_r2))
                                    if np.mean(bay_r2) > 0 else None),
            "convergence_r2_all12": ref.get("convergence_r2_all12"),
            "jackknife_r2_all12": ref.get("jackknife_r2_all12"),
            "neural_r2_per_species": {c: float(nn_r2[i]) for i, c in enumerate(TARGET_COLUMNS)},
            "bayes_r2_per_species": {c: float(bay_r2[i]) for i, c in enumerate(TARGET_COLUMNS)},
        }

        # Per-species verdict, only where the reference is actually a posterior.
        if resolved:
            for i, c in enumerate(TARGET_COLUMNS):
                gap = float(bay_r2[i] - nn_r2[i])
                verdict_rows.append([
                    f"{a:g}", c, round(float(nn_r2[i]), 3), round(float(bay_r2[i]), 3),
                    round(gap, 3),
                    "information-limited" if bay_r2[i] <= TIE_DEX else
                    ("at ceiling" if gap <= TIE_DEX else "capacity-limited"),
                ])

    best = max((a for a in want if by_alpha[a]["resolved"]), default=None)
    head = {}
    if best is not None:
        p = payload[f"alpha_{best:g}"]
        head = {
            "resolved_alphas": [a for a in want if by_alpha[a]["resolved"]],
            f"neural_R2_all12@a{best:g}": p["neural_r2_all12"],
            f"BAYES_ceiling_R2_all12@a{best:g}": p["bayes_r2_all12"],
            "fraction_of_ceiling_reached": p["fraction_of_ceiling"],
            "excess_MSE_vs_Bayes(dex²)": p["excess_mse_dex2"],
        }

    caveats = [
        f"Reference = T0 importance sampling over {doc['n_library']} INARA training "
        f"spectra, {doc['n_test']} test planets, noise seeds {doc['noise_seeds']} "
        f"(first = TEST_NOISE_SEED, identical realizations to Section A).",
        "The ceiling is a LOWER bound: with a finite library the estimate is biased low "
        "and still rising with library size (see convergence_r2_all12). The true ceiling "
        "is at least the quoted value.",
        f"ESS gate = {L.ESS_MIN:g}. Rows marked 'NO (1-NN)' are NOT ceilings — there the "
        "weights have collapsed onto a single library member and the number is the "
        "library's nearest-neighbour density, not a posterior. The high-α / noiseless "
        "reference regime is therefore NOT bounded by this section; that needs T2.",
        "Error bars: the ordinary test-planet bootstrap understates uncertainty because "
        "every planet shares one library. jackknife_r2_all12 (independent library "
        "halves) is the honest spread.",
        "Exact for the data the models are scored on — a Gaussian contrast likelihood "
        "with known σ, which is precisely what common.observable.inject_noise "
        "implements. NOT a claim about a real telescope (correlated systematics, "
        "imperfect starlight subtraction).",
    ]
    if not any_resolved:
        caveats.insert(0, "⚠ NO α in this run had a resolved posterior — nothing here is "
                          "a ceiling. Re-run run_prior_is with lower α.")

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label,
        status="ok" if any_resolved else "preliminary",
        epistemic=EPISTEMIC, headline=head,
        tables=[
            {"name": "Neural vs exact Bayesian posterior (same planets, same α, same noise)",
             "columns": ["α", "ESS med", "resolved?", "NN R²cov", "Bayes R²cov",
                         "NN R²all12", "Bayes R²all12", "excess MSE (dex²)"],
             "rows": rows},
            {"name": "Per-species verdict (resolved α only)",
             "columns": ["α", "species", "NN R²", "Bayes R²", "gap", "verdict"],
             "rows": verdict_rows},
        ],
        caveats=caveats, provenance=ctx.provenance(),
        data={"reference": {k: v for k, v in doc.items() if k != "by_alpha"},
              "by_alpha": payload},
    )
    return core.write_result(ctx, res)


def _r2(truth_log, pred_log):
    ss = ((truth_log - pred_log) ** 2).sum(0)
    st = ((truth_log - truth_log.mean(0)) ** 2).sum(0)
    return 1.0 - ss / np.clip(st, 1e-30, None)
