"""CLI: compute the T0 exact-prior posterior sweep and cache it for Section K.

Track-INDEPENDENT — the ceiling is a property of the data, not of any checkpoint — so
the output lives alongside ``results_v2/baselines/`` rather than under a track, and
Section K reads it for every track it reports.

    PYTHONPATH=src python -m evaluation.bayes.run_prior_is --n-test 2000 --seeds 202 303 404

Writes results_v2/bayes_reference/:
    summary.json          per-α R², ESS stats, convergence curve, jackknife, self-tests
    prior_is_a{α}.npz     post_mean_log, truth_log, idx, ess (+ samples_log if asked)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from common.pipeline import TEST_NOISE_SEED                       # noqa: E402
from evaluation.bayes import prior_is as P                        # noqa: E402
from evaluation.bayes import likelihood as L                      # noqa: E402

OUT_DIR = os.path.join(REPO, "results_v2", "bayes_reference")
DEFAULT_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 300.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-v2", default=os.path.join(REPO, "data", "cache_v2"))
    ap.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    ap.add_argument("--n-test", type=int, default=2000,
                    help="test planets; None-equivalent is 0 meaning the full split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[TEST_NOISE_SEED],
                    help="noise realizations; TEST_NOISE_SEED must be first to stay "
                         "comparable to Section A")
    ap.add_argument("--n-samples", type=int, default=0,
                    help="SIR draws per planet for the calibration battery (0 = skip)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--skip-self-test", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    st = None
    if not a.skip_self_test:
        st = P.self_test(a.cache_v2)
        print("self-test:", json.dumps(st, indent=1), flush=True)
        if not st["pass"]:
            raise SystemExit("SELF-TEST FAILED — refusing to produce T0 numbers. "
                             "Truth-in-proposal or prior-recovery is broken; the "
                             "tensor/label alignment or the σ convention is wrong.")

    res = P.run(a.cache_v2, alphas=tuple(a.alphas), n_test=(a.n_test or None),
                noise_seeds=tuple(a.seeds), n_samples=a.n_samples)
    summ = P.summarize(res)

    for alpha, r in res.items():
        payload = {k: r[k] for k in ("post_mean_log", "truth_log", "idx", "ess", "ess_mask")}
        if "samples_log" in r:
            payload["samples_log"] = r["samples_log"]
        np.savez_compressed(os.path.join(a.out, f"prior_is_a{alpha:g}.npz"), **payload)
        # ESS-gated subset — a DIAGNOSTIC, never the headline.
        #
        # Gating on per-planet ESS selects planets whose likelihood is flat, and those
        # are systematically the planets with LESS spread-out compositions. Measured on
        # the α=1 smoke run: per-planet RMSE is essentially independent of ESS (0.415
        # dex gated vs 0.422 ungated, Spearman +0.09), but Var(truth) is ~25% lower in
        # the gated subset, so R² = 1 − MSE/Var falls from 0.096 to 0.029 purely through
        # the denominator. Quoting the gated R² as "the conservative number" would
        # therefore understate the ceiling for a reason that has nothing to do with
        # estimator validity.
        #
        # RMSE is denominator-free and IS comparable across subsets, so it is reported
        # for both. The ESS gate's real load-bearing use is at the α level ("resolved"),
        # not per-planet.
        m = r["ess_mask"]
        summ[alpha]["n_gated"] = int(m.sum())
        summ[alpha]["rmse_all12_dex"] = float(
            np.sqrt(((r["truth_log"] - r["post_mean_log"]) ** 2).mean()))
        if m.sum() >= 30:
            g = P.r2_log10(r["truth_log"][m], r["post_mean_log"][m])
            summ[alpha]["r2_all12_gated"] = float(np.mean(g))
            summ[alpha]["r2_covered_gated"] = float(np.mean(
                [g[i] for i, c in enumerate(P.TARGET_COLUMNS)
                 if c in ("H2O", "CO2", "O2", "CH4", "O3")]))
            summ[alpha]["rmse_all12_dex_gated"] = float(np.sqrt(
                ((r["truth_log"][m] - r["post_mean_log"][m]) ** 2).mean()))
            summ[alpha]["gated_selection_note"] = (
                "R² on this subset is NOT comparable to the all-planet R² (different "
                "Var(truth)); compare rmse_all12_dex_gated vs rmse_all12_dex instead.")
        if "convergence" in r:
            summ[alpha]["convergence_r2_all12"] = {
                str(k): float(np.mean(v)) for k, v in r["convergence"].items()}
        if "jackknife" in r:
            summ[alpha]["jackknife_r2_all12"] = [float(np.mean(v)) for v in r["jackknife"]]

    doc = {
        "method": "T0 prior importance sampling (exact posterior under the INARA prior)",
        "ess_min": L.ESS_MIN,
        "noise_seeds": list(a.seeds),
        "n_test": int(next(iter(res.values()))["n_test"]),
        "n_library": int(next(iter(res.values()))["n_library"]),
        "self_test": st,
        "elapsed_s": round(time.time() - t0, 1),
        "by_alpha": summ,
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(doc, f, indent=1)

    print(f"\n{'alpha':>7} {'ESSmed':>10} {'frac ok':>8} {'resolved':>9} "
          f"{'R2_cov':>8} {'R2_all':>8} {'RMSE_dex':>9} {'seed sd':>8}")
    for al in sorted(summ):
        s = summ[al]
        sd = s.get("r2_seed_std_all12")
        print(f"{al:>7g} {s['ess_median']:>10.1f} {s['ess_frac_ok']:>8.2f} "
              f"{str(s['resolved']):>9} {s['r2_covered']:>8.3f} {s['r2_all12']:>8.3f} "
              f"{s['rmse_all12_dex']:>9.3f} "
              f"{(f'{sd:.4f}' if sd is not None else '-'):>8}")
    print("\nR2/RMSE are the PRIMARY noise seed only; 'seed sd' is the spread of "
          "R2_all12 across seeds.\nRows with resolved=False are 1-NN readouts, not "
          "ceilings. All values are LOWER bounds\n(finite library) — see "
          "convergence_r2_all12 in summary.json for whether they are still rising.")
    print(f"\nwrote {a.out}  ({doc['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
