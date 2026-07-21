"""How much does the Sagan catalog's missing blue half cost us?

The catalog stops at 0.45 um. The model's input grid starts at 0.20 um -- 35% of its
4378 bins, and exactly where O3's Hartley (0.20-0.31) and Huggins (0.31-0.35) bands and
the steepest Rayleigh slope live. Every Sagan body is therefore fed a spectrum whose
blue third is a flat extrapolation.

Rather than hand-wave that caveat, we MEASURE it: take INARA test planets (truth known),
apply the *same* mutilation, and score. The resulting per-gas R2 drop is the error bar on
every number in run_sagan.py.

A second axis matters just as much. The catalog's bodies span R~12 (Io, 30 points) to
R~870 (Earth, 1752 points), while the model trained at R~1900. Venus and Mars -- two of
the four terrestrials -- are effectively broadband photometry. So we also degrade
resolution and cross the two factors.

    PYTHONPATH=src:. python3 sagan_eval/bluegap.py --track causal --n 2000

Read-only: loads checkpoints via evaluation.core, mutates nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))

from evaluation import core                                  # noqa: E402
from common.data import TARGET_COLUMNS                       # noqa: E402
from common.pipeline import load_eval_raw                    # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE_EDGE = 0.45          # catalog blue cutoff, um
COVERED = core.COVERED_DEFAULT


def grid_trimmed():
    from sagan_eval.ingest import inara_grid
    from common.data import TRIM_TAIL_BINS
    g = inara_grid()
    return g[:-TRIM_TAIL_BINS] if TRIM_TAIL_BINS else g


# --------------------------------------------------------------------------- #
# Mutilations, applied in CONTRAST space so they mean the same thing as the
# operation ingest.py performs on a catalog albedo (C is linear in Ag).
# --------------------------------------------------------------------------- #
def _split(raw_x):
    """raw_x[N,2,L] = [star+planet, planet] -> (star, contrast)."""
    sp, pl = raw_x[:, 0], raw_x[:, 1]
    star = sp - pl
    contrast = pl / sp.clamp_min(1e-300)
    return star, contrast


def _rejoin(star, contrast):
    contrast = contrast.clamp(0.0, 0.999999)
    planet = contrast / (1.0 - contrast) * star
    return torch.stack([star + planet, planet], dim=1)


def blue_fill(raw_x, wl, edge=BLUE_EDGE):
    """Hold the contrast flat blueward of ``edge`` at its value just redward of it.

    Matches np.interp's clamping in ingest.albedo_on_grid. The seam value is a median
    over a narrow window so a single noisy bin cannot set the whole blue third.
    """
    star, C = _split(raw_x)
    below = wl < edge
    seam = (wl >= edge) & (wl < edge + 0.01)
    fill = C[:, seam].median(dim=1, keepdim=True).values
    C = C.clone()
    C[:, below] = fill.expand(-1, int(below.sum()))
    return _rejoin(star, C)


def degrade_resolution(raw_x, wl, R):
    """Gaussian-smooth the contrast to resolving power R (constant in ln-lambda)."""
    from scipy.ndimage import gaussian_filter1d
    star, C = _split(raw_x)
    lnw = np.log(wl)
    uni = np.linspace(lnw[0], lnw[-1], len(lnw))
    d = uni[1] - uni[0]
    sigma_pix = (1.0 / (R * 2.3548)) / d          # FWHM = lambda/R  ->  sigma in ln-lambda
    Cn = C.numpy()
    out = np.empty_like(Cn)
    for i in range(Cn.shape[0]):
        s = np.interp(uni, lnw, Cn[i])
        s = gaussian_filter1d(s, sigma_pix, mode="nearest")
        out[i] = np.interp(lnw, uni, s)
    return _rejoin(star, torch.from_numpy(out).float())


VARIANTS = {
    "full":                 lambda rx, wl: rx,
    "bluefill":             lambda rx, wl: blue_fill(rx, wl),
    "lowres_R870":          lambda rx, wl: degrade_resolution(rx, wl, 870),   # Earth/Titan/giants
    "lowres_R22":           lambda rx, wl: degrade_resolution(rx, wl, 22),    # Venus/Mars
    "bluefill+lowres_R870": lambda rx, wl: blue_fill(degrade_resolution(rx, wl, 870), wl),
    "bluefill+lowres_R22":  lambda rx, wl: blue_fill(degrade_resolution(rx, wl, 22), wl),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="causal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    ctx = core.EvalContext(track=a.track, seeds=a.seeds, suffix=a.suffix)
    if not ctx.has_checkpoints:
        sys.exit(f"no checkpoints for {a.track} seeds={a.seeds}")

    _, _, lt = ctx.ckpt_config()
    raw_x, y_lin, noise, _, _ = load_eval_raw(ctx.cache_v2, "test", lt)
    raw_x, y_lin, noise = raw_x[:a.n], y_lin[:a.n], noise[:a.n]
    wl = grid_trimmed()
    assert len(wl) == raw_x.shape[-1], (len(wl), raw_x.shape)
    y_true = y_lin.numpy()

    print(f"track={ctx.label}  seeds={ctx.seeds}  N={raw_x.shape[0]}  L={raw_x.shape[-1]}")
    print(f"blue bins (<{BLUE_EDGE} um): {(wl < BLUE_EDGE).sum()}/{len(wl)} "
          f"= {100*(wl < BLUE_EDGE).mean():.1f}%\n")

    gases = COVERED
    gi = [TARGET_COLUMNS.index(g) for g in gases]
    rows, results = [], {}
    for name, fn in VARIANTS.items():
        rx = fn(raw_x.clone(), wl)
        out = core.predict_raw(ctx, rx, noise, noiseless=True)
        sc = core.score(y_true, out["ens"], covered=gases, bootstrap=0)
        results[name] = {"r2_covered": float(sc["r2_covered"]),
                         "per_gas": {g: float(sc["per_species"][g]["r2"]) for g in gases}}
        rows.append([name, results[name]["r2_covered"]] + [results[name]["per_gas"][g] for g in gases])

    hdr = f"{'variant':<24}{'R2_cov':>9}" + "".join(f"{g:>9}" for g in gases)
    print(hdr); print("-" * len(hdr))
    base = results["full"]["r2_covered"]
    for r in rows:
        print(f"{r[0]:<24}{r[1]:>9.3f}" + "".join(f"{v:>9.3f}" for v in r[2:]))
    print()
    print(f"{'PENALTY vs full':<24}{'dR2_cov':>9}" + "".join(f"{g:>9}" for g in gases))
    print("-" * len(hdr))
    for name in VARIANTS:
        if name == "full":
            continue
        d = results[name]["r2_covered"] - base
        dg = [results[name]["per_gas"][g] - results["full"]["per_gas"][g] for g in gases]
        print(f"{name:<24}{d:>+9.3f}" + "".join(f"{v:>+9.3f}" for v in dg))

    path = os.path.join(OUT, f"bluegap_{ctx.label}.json")
    with open(path, "w") as f:
        json.dump({"track": ctx.label, "seeds": ctx.seeds, "n": int(raw_x.shape[0]),
                   "blue_edge_um": BLUE_EDGE, "gases": gases, "results": results}, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
