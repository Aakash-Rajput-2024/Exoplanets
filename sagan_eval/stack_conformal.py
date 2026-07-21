"""Cross-architecture stacking + conformal validity — both untried in this repo.

STACK: per-gas linear stack (log10) of all 6 frozen tracks, fit on half of INARA test,
scored on the other half. Zero retraining; the only "training" is a 6-coefficient
least-squares per gas. Compares vs best single track and plain ensemble mean.

CONFORMAL: split-conformal per-gas intervals (q68/q90 of |log10 residual| on the fit
half) from the causal track; check empirical coverage on the eval half at native
resolution and after degrading to R=22. If coverage collapses under resolution shift,
the intervals -- like any fixed uncertainty -- are only valid in-distribution.

    PYTHONPATH=src:. python3 sagan_eval/stack_conformal.py --n 1200
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
from evaluation import core                                    # noqa: E402
from common.pipeline import load_eval_raw                      # noqa: E402
from common.data import TARGET_COLUMNS                         # noqa: E402
from sagan_eval.bluegap import degrade_resolution, grid_trimmed  # noqa: E402

GASES = core.COVERED_DEFAULT
TRACKS = [("original1dcnn", [0], ""), ("optimized1dcnn", [0, 1, 2], ""),
          ("transformerarch", [0], ""), ("causal", [0], ""),
          ("causal_cfi", [0, 1], "_cfi"), ("causal_xl", [0], "_cfi")]
LOG_FLOOR = 1e-12


def l10(x):
    return np.log10(np.clip(x, LOG_FLOOR, 1))


def r2(ly, lp):
    ss = ((ly - ly.mean()) ** 2).sum()
    return 1.0 - ((ly - lp) ** 2).sum() / ss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200)
    a = ap.parse_args()
    half = a.n // 2
    gi = {g: TARGET_COLUMNS.index(g) for g in GASES}

    ctx0 = core.EvalContext(track="causal", seeds=[0])
    _, _, lt = ctx0.ckpt_config()
    raw, ylin, noise, _, _ = load_eval_raw(ctx0.cache_v2, "test", lt)
    raw, y, noise = raw[:a.n], ylin[:a.n].numpy(), noise[:a.n]

    preds, singles = {}, {}
    for t, seeds, suf in TRACKS:
        ctx = core.EvalContext(track=t, seeds=seeds, suffix=suf)
        if not ctx.has_checkpoints:
            continue
        preds[ctx.label] = core.predict_raw(ctx, raw, noise, noiseless=True)["ens"]
        singles[ctx.label] = np.mean([r2(l10(y[half:, gi[g]]), l10(preds[ctx.label][half:, gi[g]]))
                                      for g in GASES])
        core.free_device(ctx.device)
    labels = list(preds)
    print("single-track R2_covered (eval half):")
    for k, v in sorted(singles.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<18}{v:+.3f}")

    # plain mean + per-gas stack
    mean_p = np.mean([preds[k] for k in labels], axis=0)
    mean_r2 = np.mean([r2(l10(y[half:, gi[g]]), l10(mean_p[half:, gi[g]])) for g in GASES])
    stack_r2 = {}
    for g in GASES:
        i = gi[g]
        X = np.stack([l10(preds[k][:, i]) for k in labels], axis=1)
        Xf, Xe = X[:half], X[half:]
        yf, ye = l10(y[:half, i]), l10(y[half:, i])
        A = np.column_stack([Xf, np.ones(half)])
        w, *_ = np.linalg.lstsq(A, yf, rcond=None)
        pe = np.column_stack([Xe, np.ones(a.n - half)]) @ w
        stack_r2[g] = r2(ye, pe)
    print(f"\nensemble mean (6 tracks)   R2_covered = {mean_r2:+.3f}")
    print(f"per-gas linear stack       R2_covered = {np.mean(list(stack_r2.values())):+.3f}"
          f"   per gas: " + " ".join(f"{g}={stack_r2[g]:+.2f}" for g in GASES))
    best = max(singles.values())
    print(f"best single                R2_covered = {best:+.3f}"
          f"   -> stack gain {np.mean(list(stack_r2.values())) - best:+.3f}")

    # ---- conformal (causal) ------------------------------------------------------
    print("\nsplit-conformal (causal, |log10 residual| quantiles from fit half):")
    p_nat = preds["causal"]
    wl = grid_trimmed()
    raw22 = degrade_resolution(raw[half:].clone(), wl, 22)
    p22 = core.predict_raw(ctx0, raw22, noise[half:], noiseless=True)["ens"]
    print(f"{'gas':<6}{'q68 width(dex)':>15}{'cov68 native':>14}{'cov68 @R22':>12}{'q68 width @R22 needed':>23}")
    for g in GASES:
        i = gi[g]
        res_f = np.abs(l10(y[:half, i]) - l10(p_nat[:half, i]))
        q68 = np.quantile(res_f, 0.68)
        cov_nat = float(np.mean(np.abs(l10(y[half:, i]) - l10(p_nat[half:, i])) <= q68))
        res22 = np.abs(l10(y[half:, i]) - l10(p22[:, i]))
        cov22 = float(np.mean(res22 <= q68))
        need = np.quantile(res22, 0.68)
        print(f"{g:<6}{q68:>15.2f}{cov_nat:>14.2f}{cov22:>12.2f}{need:>23.2f}")
    print("\n(valid intervals require cov68 ~ 0.68; @R22 the same intervals under-cover and the")
    print(" honest width is the last column — resolution-conditioned conformal, no retraining)")


if __name__ == "__main__":
    main()
