"""Turn sagan_results.json into the three numbers that decide whether any of it means
anything.

1. PRIOR COLLAPSE. If a model assigns the same dominant gas to all 19 bodies, its
   "detection" on any one of them is a constant, not a measurement. Reported as the
   fraction of bodies sharing the modal dominant gas.

2. MATCHED-RESOLUTION DETECTION AUROC. The catalog gives us, for the first time, real
   spectra of bodies that certainly DO and certainly DO NOT contain a gas. Per gas:
       positives = bodies whose true VMR > 1%   (Earth for O2; the 4 giants + Titan
                                                 for CH4; Venus + Mars for CO2)
       negatives = the airless bodies, whose true VMR is exactly 0
   AUROC of the predicted abundance over positives-vs-negatives is a real detection
   score on real data. 0.5 = coin flip.

   Resolution is matched wherever possible, because bluegap.py showed R~22 sampling
   costs -0.7 to -1.1 R2. The CH4 test is the clean one: all 5 positives and 5 of the
   negatives sit at R = 829-867.

3. EARTH vs MOON. Lundock observed both on 2008-11-21, same instrument, same telluric
   division, both R=867. One has 21% O2. The other is a rock. Any gap in predicted O2
   between them is the model's entire real-data oxygen sensitivity.

Reference scale: the INARA test-label prior for O2+O3, so "0.45" can be read as high or
low rather than floating free.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))

from common.data import TARGET_COLUMNS                 # noqa: E402
from sagan_eval.ingest import BODY_CLASS               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R_HI = 400.0

# True VMR > 1% for these bodies (from benchmarks/solar_system_truth.json + Titan GCMS).
POSITIVES = {
    "O2":  ["Earth"],
    "CH4": ["Titan", "Jupiter", "Saturn", "Uranus", "Neptune"],
    "CO2": ["Venus", "Mars"],
}
AIRLESS = [b for b, c in BODY_CLASS.items() if c == "airless"]


def auroc(pos, neg):
    """Mann-Whitney AUROC with tie correction. pos/neg are 1-D score arrays."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def inara_o2o3_prior(cache_v2=None):
    """Mean/median O2+O3 in the INARA test LABELS — the scale to read predictions against."""
    import torch
    cache_v2 = cache_v2 or os.path.join(REPO, "data", "cache_v2")
    y = torch.load(os.path.join(cache_v2, "test_y.pt"), map_location="cpu").numpy()
    s = y[:, TARGET_COLUMNS.index("O2")] + y[:, TARGET_COLUMNS.index("O3")]
    return float(s.mean()), float(np.median(s))


def analyze_track(rows, gases=("O2", "CH4", "CO2")):
    by = {r["body"]: r for r in rows}
    doms = [r["dominant_pred"] for r in rows]
    modal = max(set(doms), key=doms.count)

    out = {"modal_dominant": modal, "collapse_frac": doms.count(modal) / len(doms),
           "n_distinct_dominant": len(set(doms))}

    for cls in ("terrestrial", "giant", "airless"):
        v = [r["o2_plus_o3"] for r in rows if r["body_class"] == cls]
        out[f"o2o3_{cls}"] = float(np.mean(v))

    # detection AUROC, all bodies and resolution-matched
    for g in gases:
        pos_b = [b for b in POSITIVES[g] if b in by]
        neg_b = [b for b in AIRLESS if b in by]
        p = [by[b]["pred"][g] for b in pos_b]
        n = [by[b]["pred"][g] for b in neg_b]
        out[f"auroc_{g}"] = auroc(p, n)
        ph = [by[b]["pred"][g] for b in pos_b if by[b]["native_R"] >= R_HI]
        nh = [by[b]["pred"][g] for b in neg_b if by[b]["native_R"] >= R_HI]
        out[f"auroc_{g}_hires"] = auroc(ph, nh)
        out[f"n_pos_hires_{g}"], out[f"n_neg_hires_{g}"] = len(ph), len(nh)

    e, m = by.get("Earth"), by.get("Moon")
    if e and m:
        out["earth_O2"] = e["pred"]["O2"]
        out["moon_O2"] = m["pred"]["O2"]
        out["earth_minus_moon_O2"] = e["pred"]["O2"] - m["pred"]["O2"]
        out["earth_o2o3"] = e["o2_plus_o3"]
        out["moon_o2o3"] = m["o2_plus_o3"]
        # Earth's rank in predicted O2 among all 19 (1 = highest). Ideal = 1.
        order = sorted(rows, key=lambda r: -r["pred"]["O2"])
        out["earth_rank_O2_of19"] = 1 + [r["body"] for r in order].index("Earth")
    return out


def main():
    with open(os.path.join(HERE, "sagan_results.json")) as f:
        doc = json.load(f)
    res = doc["results"]

    mean, med = inara_o2o3_prior()
    print(f"INARA test-label prior for O2+O3:  mean={mean:.3f}  median={med:.3f}")
    print("(a body with NO atmosphere should not score above this; a rock has zero O2)\n")

    for tag in ("clear", "declouded"):
        if not any(tag in r for r in res.values()):
            continue
        print("=" * 118)
        print(f"[{tag.upper()}]")
        print("=" * 118)
        h = (f"{'track':<18}{'modal':>7}{'collapse':>10}{'#dom':>6}"
             f"{'O2+O3 terr':>12}{'giant':>8}{'airless':>9}"
             f"{'AUROC_O2':>10}{'AUROC_CH4':>11}{'CH4 hi-res':>12}{'Earth-Moon O2':>15}")
        print(h); print("-" * len(h))
        rowcache = {}
        for tr, r in res.items():
            if tag not in r:
                continue
            a = analyze_track(r[tag])
            rowcache[tr] = a
            print(f"{tr:<18}{a['modal_dominant']:>7}{a['collapse_frac']:>9.0%}"
                  f"{a['n_distinct_dominant']:>6}"
                  f"{a['o2o3_terrestrial']:>12.3f}{a['o2o3_giant']:>8.3f}{a['o2o3_airless']:>9.3f}"
                  f"{a['auroc_O2']:>10.2f}{a['auroc_CH4']:>11.2f}"
                  f"{a['auroc_CH4_hires']:>12.2f}{a['earth_minus_moon_O2']:>+15.3f}")
        print()
        nph = next(iter(rowcache.values()))
        print(f"  AUROC_CH4 hi-res uses {nph['n_pos_hires_CH4']} CH4-rich vs "
              f"{nph['n_neg_hires_CH4']} airless bodies, all R=829-867 (resolution matched).")
        print(f"  AUROC 0.5 = chance. Earth-Moon O2 > 0 means the model sees Earth's oxygen.\n")

    # Earth vs Moon detail, clear pass
    print("=" * 118)
    print("EARTH vs MOON — same night (2008-11-21), same instrument, both R=867")
    print("=" * 118)
    h = f"{'track':<18}{'pass':<11}{'Earth O2':>10}{'Moon O2':>10}{'diff':>9}{'Earth rank O2 /19':>20}"
    print(h); print("-" * len(h))
    for tr, r in res.items():
        for tag in ("clear", "declouded"):
            if tag not in r:
                continue
            a = analyze_track(r[tag])
            print(f"{tr:<18}{tag:<11}{a['earth_O2']:>10.3f}{a['moon_O2']:>10.3f}"
                  f"{a['earth_minus_moon_O2']:>+9.3f}{a['earth_rank_O2_of19']:>20d}")
    print("\ntrue O2:  Earth 0.2095   Moon 0.0 (no atmosphere)")


if __name__ == "__main__":
    main()
