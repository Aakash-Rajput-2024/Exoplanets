"""Which of the 12 target gases can the Sagan catalog actually test, and how do they score?

A gas is TESTABLE here only if some body genuinely imprints it on its spectrum. We take
positives = bodies where the gas is in the benchmark's own ``covered_species`` (the forward
model imprints it) AND its VMR exceeds 1e-3 (so Earth's 1.9 ppm CH4 and 7e-7 O3 don't count
as "methane-rich" or "ozone-rich"). Negatives = the 11 airless bodies, whose true VMR is
exactly zero for every gas.

Note VMR is the wrong variable for a giant: Jupiter's CH4 is only 0.002 by number, but the
column is optically thick and its 1.7 um band depth is 0.91 in the delivered albedo. The
covered_species ∩ VMR>1e-3 rule happens to select exactly the bodies with visible bands.

Two extra probes the catalog uniquely allows:

  N2 NULL CONTROL. Earth is 78% N2 and Titan 95% N2, but N2 has no reflected-light bands at
  all -- it appears in no covered_species list. Predicted N2 therefore MUST be uninformative.
  Its AUROC is a direct readout of the models' prior, on a gas we know is invisible.

  SURFACE-ICE CONFOUND. Water ice absorbs at 1.5 and 2.0 um -- the same place gaseous H2O
  does. Six airless moons are water-ice covered, Io is coated in SO2 frost, and Pluto's
  surface is CH4/N2/CO ice. If the models call those bodies wet/sulphurous/methane-rich, they
  are detecting the right molecule in the wrong PHASE: a physically meaningful false positive,
  not noise.

    PYTHONPATH=src:. python3 sagan_eval/gases.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
HERE = os.path.dirname(os.path.abspath(__file__))

from common.data import TARGET_COLUMNS                # noqa: E402
from sagan_eval.ingest import BODY_CLASS              # noqa: E402
from sagan_eval.analyze import auroc                  # noqa: E402

VMR_MIN = 1e-3
R_HI = 400.0
AIRLESS = [b for b, c in BODY_CLASS.items() if c == "airless"]

# Surface composition of the airless bodies (Madden & Kaltenegger Table 1 + source papers).
WATER_ICE = ["Europa", "Ganymede", "Callisto", "Enceladus", "Dione", "Rhea"]
DRY_ROCK = ["Mercury", "Moon", "Ceres"]
SO2_FROST = ["Io"]
CH4_ICE = ["Pluto"]


def truth_doc():
    with open(os.path.join(REPO, "src/evaluation/benchmarks/solar_system_truth.json")) as f:
        return json.load(f)


def positives_for(gas):
    """Bodies that genuinely imprint `gas`: covered_species AND VMR > 1e-3."""
    out = []
    for b in truth_doc()["bodies"]:
        if gas in b.get("covered_species", []) and float(b["vmr"].get(gas, 0.0)) > VMR_MIN:
            out.append(b["name"])
    return out


def main():
    with open(os.path.join(HERE, "sagan_results.json")) as f:
        doc = json.load(f)
    res = doc["results"]
    tracks = list(res)

    # ---------------------------------------------------------------- testability
    print("=" * 104)
    print("WHICH GASES CAN THIS CATALOG TEST AT ALL?")
    print("=" * 104)
    print(f"{'gas':<7}{'positives (bodies that imprint it)':<44}{'n_pos':>6}{'hi-res pos':>12}{'testable?':>13}")
    print("-" * 104)
    metas = doc["metas"]
    testable = {}
    for g in TARGET_COLUMNS:
        pos = positives_for(g)
        hi = [b for b in pos if metas[b]["native_R_at_1um"] >= R_HI]
        if not pos:
            verdict = "NO — no body"
        elif not hi:
            verdict = "NO — all R~22"
        elif len(hi) == 1:
            verdict = "weak (n=1)"
        else:
            verdict = "YES"
        testable[g] = (pos, hi, verdict)
        print(f"{g:<7}{', '.join(pos) if pos else '(none)':<44}{len(pos):>6}{len(hi):>12}{verdict:>13}")

    print("\n  N2 is absent from every covered_species list: 78% of Earth, 95% of Titan, and")
    print("  spectroscopically invisible in reflected light. It is the null control.\n")

    # ---------------------------------------------------------------- AUROC per gas
    print("=" * 104)
    print("DETECTION AUROC — positives vs the 11 airless bodies (true VMR = 0). 0.5 = chance.")
    print("=" * 104)
    gases = [g for g in TARGET_COLUMNS if testable[g][0]] + ["N2"]
    hdr = f"{'track':<18}" + "".join(f"{g:>11}" for g in gases)
    print(hdr); print("-" * len(hdr))
    tab = {}
    for t in tracks:
        rows = {r["body"]: r for r in res[t]["clear"]}
        cells, row = "", {}
        for g in gases:
            pos, hi, _ = testable[g] if g in testable and testable[g][0] else ([], [], "")
            if g == "N2":
                pos = ["Earth", "Titan"]          # abundant but invisible
            p = [rows[b]["pred"][g] for b in pos if b in rows]
            n = [rows[b]["pred"][g] for b in AIRLESS if b in rows]
            a = auroc(p, n)
            row[g] = a
            cells += f"{a:>11.2f}"
        tab[t] = row
        print(f"{t:<18}{cells}")
    print()
    med = {g: np.nanmedian([tab[t][g] for t in tracks]) for g in gases}
    print(f"{'MEDIAN':<18}" + "".join(f"{med[g]:>11.2f}" for g in gases))
    print("\n  CH4 is the only gas with several positives, all at matched resolution.")
    print("  CO2's only positives are Venus + Mars, both at R~22 where R2 goes negative.")
    print("  O2 and H2O have exactly one positive (Earth) — a rank, not a real AUROC.\n")

    # ---------------------------------------------------------------- fabrication
    print("=" * 104)
    print("FABRICATION — mean predicted VMR on the 11 AIRLESS bodies (true value: 0.000)")
    print("=" * 104)
    import torch
    y = torch.load(os.path.join(REPO, "data/cache_v2/test_y.pt"), map_location="cpu").numpy()
    prior = {g: float(y[:, i].mean()) for i, g in enumerate(TARGET_COLUMNS)}
    show = ["O2", "CO2", "N2", "H2O", "CH4", "O3", "SO2", "CO"]
    hdr = f"{'track':<18}" + "".join(f"{g:>10}" for g in show)
    print(hdr); print("-" * len(hdr))
    for t in tracks:
        rows = {r["body"]: r for r in res[t]["clear"]}
        cells = "".join(f"{np.mean([rows[b]['pred'][g] for b in AIRLESS]):>10.3f}" for g in show)
        print(f"{t:<18}{cells}")
    print("-" * len(hdr))
    print(f"{'INARA label prior':<18}" + "".join(f"{prior[g]:>10.3f}" for g in show))
    print("\n  A body with no atmosphere should not produce a confident number in ANY column.")
    print("  Compare each row to the prior: the models are reciting it, not measuring.\n")

    # ---------------------------------------------------------------- ice confound
    print("=" * 104)
    print("SURFACE-ICE CONFOUND — right molecule, wrong phase?")
    print("=" * 104)
    print(f"  water-ice moons : {', '.join(WATER_ICE)}")
    print(f"  dry rock        : {', '.join(DRY_ROCK)}")
    print(f"  SO2 frost       : {', '.join(SO2_FROST)}     CH4 ice: {', '.join(CH4_ICE)}\n")
    hdr = (f"{'track':<18}{'H2O icy':>10}{'H2O dry':>10}{'AUROC':>8}   "
           f"{'SO2 Io':>9}{'SO2 other':>11}   {'CH4 Pluto':>11}{'CH4 dry':>10}")
    print(hdr); print("-" * len(hdr))
    for t in tracks:
        rows = {r["body"]: r for r in res[t]["clear"]}
        icy = [rows[b]["pred"]["H2O"] for b in WATER_ICE]
        dry = [rows[b]["pred"]["H2O"] for b in DRY_ROCK]
        so2_io = rows["Io"]["pred"]["SO2"]
        so2_oth = np.mean([rows[b]["pred"]["SO2"] for b in AIRLESS if b != "Io"])
        ch4_pl = rows["Pluto"]["pred"]["CH4"]
        ch4_dry = np.mean([rows[b]["pred"]["CH4"] for b in DRY_ROCK])
        print(f"{t:<18}{np.mean(icy):>10.3f}{np.mean(dry):>10.3f}{auroc(icy, dry):>8.2f}   "
              f"{so2_io:>9.3f}{so2_oth:>11.3f}   {ch4_pl:>11.3f}{ch4_dry:>10.3f}")
    print("\n  Ice absorbs at 1.5/2.0 um exactly where water VAPOUR does. If AUROC(icy vs dry)")
    print("  is high, the model is reading a surface as an atmosphere.")


if __name__ == "__main__":
    main()
