"""Is the declouder broken, or is it simply out of its domain on real spectra?

run_sagan.py shows the declouded pass destroys the one real signal the models have
(matched-resolution CH4 AUROC 1.00 -> ~0.1). Three controls separate the hypotheses:

  A. DOES IT WORK AS ADVERTISED?  clear -> grey cloud -> decloud, on INARA test spectra.
     Recovery must be strongly positive. If not, the checkpoint is bad and everything
     downstream is moot.

  B. IS IT IDENTITY-SAFE ON CLEAR INPUT?  Feed already-clear INARA spectra straight
     through. A restoration front-end must not damage an unclouded spectrum, because at
     deployment you never know whether the target is clouded.

  C. WHAT DOES IT DO TO REAL SPECTRA?  Encoded-contrast statistics and CH4 band depth on
     the Sagan bodies, before vs after.

    PYTHONPATH=src:. python3 sagan_eval/decloud_check.py --track causal
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))

from evaluation import core                                  # noqa: E402
from common.pipeline import load_eval_raw                    # noqa: E402
from common.data import TARGET_COLUMNS                       # noqa: E402
from cloud_recovery.cloud_pairs import apply_family_to_raw   # noqa: E402
from sagan_eval import build_obs, run_sagan, bluegap         # noqa: E402

# CH4 has deep, broad reflected-light bands; O2 has only the narrow 0.76 um A-band.
# (band_lo, band_hi, continuum windows) in um.
BANDS = {
    "CH4_0.89": (0.86, 0.92, [(0.82, 0.85), (0.93, 0.96)]),
    "CH4_1.7":  (1.62, 1.75, [(1.55, 1.60), (1.78, 1.83)]),
    "O2_A":     (0.757, 0.772, [(0.740, 0.752), (0.777, 0.790)]),
}


def band_contrast(enc_ch0, wl, band):
    """mean(continuum) - mean(in-band), in ENCODED units. Positive = absorption seen.

    A ratio is meaningless here: ch0 is asinh-normed and crosses zero, so dividing by the
    continuum blows up. A difference is well defined and monotone in true band depth.
    """
    lo, hi, cont = band
    inb = (wl >= lo) & (wl <= hi)
    cw = np.zeros_like(wl, bool)
    for a, b in cont:
        cw |= (wl >= a) & (wl <= b)
    if not inb.any() or not cw.any():
        return np.nan
    c = np.asarray(enc_ch0)
    return c[..., cw].mean(-1) - c[..., inb].mean(-1)


def grey_cloud(raw_x, f, b=0.5):
    """The exact grey transform GreyCloudPairCollate trains on, at chosen opacity f.

    f=0 is a perfectly clear spectrum. Training only ever drew f in GREY_F_RANGE=(0.3,1.0).
    """
    from common import cloud_families
    sp, p = raw_x[:, 0], raw_x[:, 1]
    cont = torch.from_numpy(cloud_families.estimate_continuum(p.numpy())).float()
    p_cloud = (p + f * (cont - p)) * (1.0 + b * f)
    return torch.stack([(sp - p) + p_cloud, p_cloud], dim=1)


def r2_cov(ctx, x_enc, y_true, lp):
    _, ens = core._decode_predict(ctx, torch.as_tensor(x_enc, dtype=torch.float32), lp)
    return core.score(y_true, ens, bootstrap=0)["r2_covered"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="causal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()

    ctx = core.EvalContext(track=a.track, seeds=a.seeds, suffix=a.suffix)
    dev = ctx.device
    dec, _ = run_sagan.load_declouder(dev)
    _, _, lt = ctx.ckpt_config()
    lp = core.inara_label_pipeline(ctx.cache_v2, lt)
    wl = bluegap.grid_trimmed()

    raw_x, y_lin, noise, _, _ = load_eval_raw(ctx.cache_v2, "test", lt)
    raw_x, y_lin, noise = raw_x[:a.n], y_lin[:a.n], noise[:a.n]
    y = y_lin.numpy()

    print(f"track={ctx.label}  N={a.n}\n")

    # ---- A/B. opacity sweep: where does the declouder help, where does it hurt? ----
    from cloud_recovery.cloud_pairs import GREY_F_RANGE
    print(f"A+B. GREY-OPACITY SWEEP  (training drew f in {GREY_F_RANGE}; f=0 is clear)")
    x_clear = core.encode_raw(ctx, raw_x, noise, noiseless=True)
    r_clear = r2_cov(ctx, x_clear, y, lp)

    h = f"   {'f':>5}{'seen?':>7}{'R2 cloudy':>12}{'R2 declouded':>15}{'recovery':>11}{'verdict':>12}"
    print(h); print("   " + "-" * (len(h) - 3))
    sweep = []
    for f in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        rc = grey_cloud(raw_x, f)
        xc = core.encode_raw(ctx, rc, noise, noiseless=True)
        xd = run_sagan.decloud_encoded(dec, xc, dev).numpy()
        r_cloud, r_dec = r2_cov(ctx, xc, y, lp), r2_cov(ctx, xd, y, lp)
        rec = (r_dec - r_cloud) / (r_clear - r_cloud) if abs(r_clear - r_cloud) > 1e-9 else np.nan
        seen = "yes" if GREY_F_RANGE[0] <= f <= GREY_F_RANGE[1] else "NO"
        verdict = "helps" if r_dec > r_cloud + 0.02 else ("HURTS" if r_dec < r_cloud - 0.02 else "~same")
        sweep.append((f, seen, r_cloud, r_dec, rec, verdict))
        print(f"   {f:>5.1f}{seen:>7}{r_cloud:>12.3f}{r_dec:>15.3f}{rec:>11.2f}{verdict:>12}")
    print(f"\n   R2 on untouched clear spectra = {r_clear:+.3f}")
    print("   The declouder assumes a cloud is present. Below the training floor f=0.3 it")
    print("   over-corrects: on a CLEAR spectrum it is a large, unconditional distortion.\n")

    # ---- C. behaviour on the real Sagan spectra ------------------------------
    print("C. REAL SPECTRA — Sagan bodies, encoded ch0 + CH4/O2 band depth")
    s_raw, s_noise, names, metas = build_obs.all_bodies_raw()
    xs = core.encode_raw(ctx, s_raw, s_noise, noiseless=True)
    xs_dec = run_sagan.decloud_encoded(dec, xs, dev).numpy()

    print(f"\n   {'body':<11}{'std ch0':>9}{'->':>3}{'std':>7}"
          + "".join(f"{k + ' bef/aft':>20}" for k in BANDS))
    print("   " + "-" * (30 + 20 * len(BANDS)))
    keep = ["Earth", "Titan", "Jupiter", "Saturn", "Uranus", "Neptune", "Moon", "Enceladus"]
    for i, n in enumerate(names):
        if n not in keep:
            continue
        b0, b1 = xs[i, 0], xs_dec[i, 0]
        cells = "".join(f"{band_contrast(b0, wl, bd):>+9.3f}/{band_contrast(b1, wl, bd):>+9.3f}"
                        for bd in BANDS.values())
        print(f"   {n:<11}{b0.std():>9.3f}{'->':>3}{b1.std():>7.3f}{cells}")
    print("\n   (encoded band contrast = continuum - band; positive = absorption seen)")

    # how far outside the training domain are these inputs?
    print(f"\n   encoded ch0 |value| — INARA test: median {np.median(np.abs(x_clear[:,0])):.3f}, "
          f"p99 {np.percentile(np.abs(x_clear[:,0]),99):.3f}")
    print(f"   encoded ch0 |value| — Sagan bodies: median {np.median(np.abs(xs[:,0])):.3f}, "
          f"p99 {np.percentile(np.abs(xs[:,0]),99):.3f}")


if __name__ == "__main__":
    main()
