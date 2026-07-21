"""Three untried inference-only probes on the frozen models + real Sagan bodies.

1. CONTINUUM-TWIN NULLING. For each body build a band-free twin (its own continuum via
   cloud_families.estimate_continuum) and score
        score(gas) = pred(body) - pred(twin).
   Bands are the ONLY difference between the pair, so the label prior cancels exactly.
   A fabricated gas (prior recitation) nulls to ~0; a real band survives.

2. OCCLUSION EVIDENCE CHECK. Blank the CH4 bands (continuum-fill) on the real spectra and
   re-run: if CH4 AUROC=1.00 really comes from CH4 wavelengths it must collapse, while
   blanking random control windows of equal width must not.

3. RULER BASELINE. AUROC of plain band-depth measured on the cleaned albedo. If the ruler
   also gets 1.00, the NN matches physics; it does not add magic.

    PYTHONPATH=src:. python3 sagan_eval/null_twin.py --track causal
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
from evaluation import core                                   # noqa: E402
from common.data import TARGET_COLUMNS                        # noqa: E402
from common import cloud_families                             # noqa: E402
from sagan_eval import build_obs, ingest                      # noqa: E402
from sagan_eval.analyze import auroc                          # noqa: E402
from sagan_eval.bluegap import grid_trimmed                   # noqa: E402
from sagan_eval.gases import AIRLESS                          # noqa: E402

CH4_POS = ["Titan", "Jupiter", "Saturn", "Uranus", "Neptune"]
HIRES_NEG = ["Moon", "Enceladus", "Dione", "Rhea", "Ceres"]
CH4_WIN = [(0.86, 0.92), (1.10, 1.20), (1.30, 1.45), (1.62, 1.80)]
CTRL_WIN = [(0.55, 0.61), (0.95, 1.05), (1.50, 1.58), (1.85, 1.95)]   # off-band, same total width


def occlude(raw, wl, wins):
    """Continuum-fill the planet channel inside the windows (star untouched)."""
    sp, p = raw[:, 0].clone(), raw[:, 1].clone()
    m = np.zeros(len(wl), bool)
    for a, b in wins:
        m |= (wl >= a) & (wl <= b)
    star = sp - p
    pn = p.numpy()
    for i in range(pn.shape[0]):
        pn[i, m] = np.interp(wl[m], wl[~m], pn[i, ~m])
    p = torch.from_numpy(pn).float()
    return torch.stack([star + p, p], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="causal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    ctx = core.EvalContext(track=a.track, seeds=a.seeds, suffix=a.suffix)
    wl = grid_trimmed()
    raw, noise, names, _ = build_obs.all_bodies_raw()
    idx = {n: i for i, n in enumerate(names)}
    gi = {g: TARGET_COLUMNS.index(g) for g in ["O2", "CH4", "CO2", "H2O", "N2"]}

    pred = core.predict_raw(ctx, raw, noise, noiseless=True)["ens"]

    # ---- 1. continuum twin ----------------------------------------------------
    cont = torch.from_numpy(cloud_families.estimate_continuum(raw[:, 1].numpy())).float()
    star = raw[:, 0] - raw[:, 1]
    raw_twin = torch.stack([star + cont, cont], dim=1)
    pred_twin = core.predict_raw(ctx, raw_twin, noise, noiseless=True)["ens"]
    score = pred - pred_twin

    print(f"track={ctx.label}\n")
    print("1. CONTINUUM-TWIN NULLING  score = pred(body) - pred(band-free twin)")
    print(f"{'':16}{'raw pred':>22}{'twin-nulled score':>26}")
    print(f"{'metric':<16}{'AUROC/val':>22}{'AUROC/val':>26}")
    print("-" * 66)
    for g, pos, neg in [("CH4", CH4_POS, HIRES_NEG), ("O2", ["Earth"], AIRLESS)]:
        j = gi[g]
        a0 = auroc([pred[idx[b], j] for b in pos], [pred[idx[b], j] for b in neg])
        a1 = auroc([score[idx[b], j] for b in pos], [score[idx[b], j] for b in neg])
        print(f"AUROC {g:<10}{a0:>22.2f}{a1:>26.2f}")
    e, m = idx["Earth"], idx["Moon"]
    j = gi["O2"]
    print(f"Earth-Moon O2   {pred[e,j]-pred[m,j]:>+22.3f}{score[e,j]-score[m,j]:>+26.3f}")
    for g in ["O2", "CO2", "N2"]:
        j = gi[g]
        r0 = np.mean([abs(pred[idx[b], j]) for b in AIRLESS])
        r1 = np.mean([abs(score[idx[b], j]) for b in AIRLESS])
        print(f"|{g}| on rocks  {r0:>22.3f}{r1:>26.3f}   (fabrication -> should shrink)")

    # ---- 2. occlusion ----------------------------------------------------------
    print("\n2. OCCLUSION — CH4 AUROC (5 CH4-rich vs 5 airless, matched R) after blanking:")
    for tag, wins in [("nothing", None), ("CH4 bands", CH4_WIN), ("control bands", CTRL_WIN)]:
        rx = raw if wins is None else occlude(raw, wl, wins)
        p = pred if wins is None else core.predict_raw(ctx, rx, noise, noiseless=True)["ens"]
        j = gi["CH4"]
        av = auroc([p[idx[b], j] for b in CH4_POS], [p[idx[b], j] for b in HIRES_NEG])
        print(f"   blank {tag:<14} AUROC = {av:.2f}")
        core.free_device(ctx.device)

    # ---- 3. ruler ---------------------------------------------------------------
    grid = ingest.inara_grid()
    depths = {}
    for b in CH4_POS + HIRES_NEG:
        ag, _ = ingest.albedo_on_grid(ingest.CANONICAL[b], grid)
        d = 0.0
        for lo, hi in [(0.86, 0.92), (1.62, 1.75)]:
            inb = (grid >= lo) & (grid <= hi)
            cw = ((grid >= lo - 0.05) & (grid < lo)) | ((grid > hi) & (grid <= hi + 0.05))
            d += 1.0 - ag[inb].mean() / max(ag[cw].mean(), 1e-9)
        depths[b] = d
    av = auroc([depths[b] for b in CH4_POS], [depths[b] for b in HIRES_NEG])
    print(f"\n3. RULER — band-depth-only CH4 AUROC = {av:.2f}  (NN must at least match this)")


if __name__ == "__main__":
    main()
