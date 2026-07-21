"""Per-gas resolution requirements + can affine recalibration rescue low-R?

Sweep R on INARA test (truth known), score per gas; then at each R fit a per-gas affine
map (log10 pred -> truth) on a held-in half and score held-out — if low-R failure is a
calibratable bias, affine rescues it; if information is gone, nothing can.

    PYTHONPATH=src:. python3 sagan_eval/resolution_curve.py --track causal --n 1200
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
from evaluation import core                          # noqa: E402
from common.pipeline import load_eval_raw            # noqa: E402
from common.data import TARGET_COLUMNS               # noqa: E402
from sagan_eval.bluegap import degrade_resolution, grid_trimmed   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GASES = core.COVERED_DEFAULT
LOG_FLOOR = 1e-12


def r2_log(y, p):
    ly, lp = np.log10(np.clip(y, LOG_FLOOR, 1)), np.log10(np.clip(p, LOG_FLOOR, 1))
    ss = ((ly - ly.mean()) ** 2).sum()
    return 1.0 - ((ly - lp) ** 2).sum() / ss if ss > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="causal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=1200)
    a = ap.parse_args()

    ctx = core.EvalContext(track=a.track, seeds=a.seeds, suffix=a.suffix)
    _, _, lt = ctx.ckpt_config()
    raw, ylin, noise, _, _ = load_eval_raw(ctx.cache_v2, "test", lt)
    raw, y, noise = raw[:a.n], ylin[:a.n].numpy(), noise[:a.n]
    wl = grid_trimmed()
    gi = {g: TARGET_COLUMNS.index(g) for g in GASES}
    half = a.n // 2

    Rs = [22, 50, 100, 200, 400, 870, None]        # None = native
    out = {}
    print(f"track={ctx.label}  N={a.n}  (fit half={half}, eval half={a.n-half})\n")
    h = f"{'R':>7} " + "".join(f"{g:>8}" for g in GASES) + "   |  affine-rescued: " + " ".join(f"{g:>7}" for g in GASES)
    print(h); print("-" * len(h))
    for R in Rs:
        rx = raw if R is None else degrade_resolution(raw.clone(), wl, R)
        ens = core.predict_raw(ctx, rx, noise, noiseless=True)["ens"]
        raw_r2, fix_r2 = {}, {}
        for g in GASES:
            i = gi[g]
            yt, pp = y[half:, i], ens[half:, i]
            raw_r2[g] = r2_log(yt, pp)
            # affine in log10 fit on first half
            lp_f = np.log10(np.clip(ens[:half, i], LOG_FLOOR, 1))
            ly_f = np.log10(np.clip(y[:half, i], LOG_FLOOR, 1))
            A = np.polyfit(lp_f, ly_f, 1)
            lp_e = np.polyval(A, np.log10(np.clip(pp, LOG_FLOOR, 1)))
            ly_e = np.log10(np.clip(yt, LOG_FLOOR, 1))
            ss = ((ly_e - ly_e.mean()) ** 2).sum()
            fix_r2[g] = 1.0 - ((ly_e - lp_e) ** 2).sum() / ss
        tag = "native" if R is None else str(R)
        out[tag] = {"raw": raw_r2, "affine": fix_r2}
        print(f"{tag:>7} " + "".join(f"{raw_r2[g]:>8.2f}" for g in GASES)
              + "   |                    " + " ".join(f"{fix_r2[g]:>7.2f}" for g in GASES))
        core.free_device(ctx.device)

    nat = out["native"]["raw"]
    print("\nminimum R for R2 >= 0.5 x native (raw | affine-rescued):")
    for g in GASES:
        thr = 0.5 * nat[g]
        mr = next((t for t in ["22", "50", "100", "200", "400", "870"] if out[t]["raw"][g] >= thr), ">870")
        ma = next((t for t in ["22", "50", "100", "200", "400", "870"] if out[t]["affine"][g] >= thr), ">870")
        print(f"  {g:<5} raw: R>={mr:<6} affine: R>={ma}")

    with open(os.path.join(HERE, f"resolution_curve_{ctx.label}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
