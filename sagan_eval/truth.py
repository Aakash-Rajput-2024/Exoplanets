"""Ground truth for the 19 Sagan-catalog bodies, in three epistemic classes.

evaluation/benchmarks/solar_system_truth.json already carries 8 bodies (4 terrestrials +
4 giants). The catalog adds 11 more, and they are the scientifically interesting ones:

  terrestrial (4)  Earth, Venus, Mars, Titan
      Substantial atmosphere, composition inside the 12-species simplex.
      -> ACCURACY: dominant-gas correct, dex error on covered species.

  giant (4)  Jupiter, Saturn, Uranus, Neptune
      H2/He dominated; the true mass sits OUTSIDE the simplex, so no dex error is
      definable. -> HONESTY: must not fabricate high O2/O3.

  airless (11)  Mercury, Moon, Io, Europa, Ganymede, Callisto, Enceladus, Dione,
                Rhea, Ceres, Pluto
      NO atmosphere. The reflected spectrum is pure surface mineralogy/ice.
      The simplex is forced to sum to 1, so the model MUST output some composition --
      there is no "none of the above" head. -> FALSE-POSITIVE PROBE.

The airless class is the control the pipeline never had. EVAL_REPORT records that every
model calls real Earth O2-dominant. If a model also calls the MOON O2-dominant, then that
Earth "detection" carries no information about oxygen -- it is what the network says to
any grey-ish reflectance slope. Nothing else in the repo can separate those two
hypotheses, because every other Section-E spectrum is generated from a gas mixture.

Pluto is grouped airless: its 1e-5 bar N2/CH4 atmosphere imprints no reflected-light
band depth at these wavelengths (surface CH4/N2 ice does, but that is a surface, not a
column abundance the model is asked to retrieve).
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))

from evaluation import known_targets                 # noqa: E402
from common.data import TARGET_COLUMNS               # noqa: E402
from sagan_eval.ingest import BODY_CLASS, CLOUDY     # noqa: E402

O2O3 = [TARGET_COLUMNS.index("O2"), TARGET_COLUMNS.index("O3")]

# Dominant surface material -- what the spectrum actually encodes for an airless body.
# Sources: Madden & Kaltenegger 2018 Table 1 and the underlying observation papers.
AIRLESS_SURFACE = {
    "Mercury":   "silicate regolith (dark, red slope)",
    "Moon":      "anorthosite/basalt regolith",
    "Io":        "SO2 frost + sulfur allotropes",
    "Europa":    "water ice + hydrated salts",
    "Ganymede":  "water ice + silicate",
    "Callisto":  "water ice + dark carbonaceous",
    "Enceladus": "fresh water ice (Ag~1)",
    "Dione":     "water ice",
    "Rhea":      "water ice",
    "Ceres":     "carbonaceous, hydrated minerals",
    "Pluto":     "N2/CH4/CO surface ices",
}

# Resolution trust, set by bluegap.py: at R~870 the covered-species R2 penalty is
# -0.03; at R~22 it is -1.11 (R2 goes negative). Anything sampled near R~22 cannot
# support a quantitative abundance claim, only a qualitative one.
R_TRUST_HI = 400.0


def _covered_for(body, spec):
    return spec.get("covered_species", []) if spec else []


def all_targets(metas=None):
    """[{name, body_class, true_vector|None, covered_species, ...}] for all 19 bodies.

    ``metas`` = ingest.build_all()'s meta dict, used to attach the native sampling
    resolution and the derived trust flag.
    """
    known = {t["name"]: t for t in known_targets.solar_system_targets()}
    out = []
    for body, cls in BODY_CLASS.items():
        spec = known.get(body)
        m = (metas or {}).get(body, {})
        R = m.get("native_R_at_1um", np.nan)
        out.append({
            "name": body,
            "body_class": cls,
            "true_vector": spec["true_vector"] if spec else None,
            "covered_species": _covered_for(body, spec),
            "representable": bool(spec["representable"]) if spec else False,
            "surface": AIRLESS_SURFACE.get(body, ""),
            "cloudy": CLOUDY.get(body, ""),
            "native_R": float(R),
            "quantitative": bool(R >= R_TRUST_HI),   # else qualitative only
            "note": spec.get("note", "") if spec else "no atmosphere",
        })
    return out


def o2_o3_sum(pred_lin):
    return float(np.asarray(pred_lin)[O2O3].sum())


def dominant(pred_lin):
    return TARGET_COLUMNS[int(np.argmax(np.asarray(pred_lin)))]


if __name__ == "__main__":
    from sagan_eval.ingest import build_all
    metas = {b: m for b, (_, m) in build_all().items()}
    hdr = f"{'body':<11}{'class':<13}{'R@1um':>7}{'quant?':>8}  {'truth / surface'}"
    print(hdr); print("-" * 88)
    for t in all_targets(metas):
        what = (f"dominant={TARGET_COLUMNS[int(np.argmax(t['true_vector']))]}"
                if t["true_vector"] is not None else t["surface"])
        print(f"{t['name']:<11}{t['body_class']:<13}{t['native_R']:>7.0f}"
              f"{'yes' if t['quantitative'] else 'NO':>8}  {what}")
    n = {c: sum(1 for t in all_targets(metas) if t['body_class'] == c)
         for c in ("terrestrial", "giant", "airless")}
    print(f"\ncounts: {n}   quantitative-resolution bodies: "
          f"{sum(t['quantitative'] for t in all_targets(metas))}/19")
