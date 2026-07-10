"""Load the bundled credible-source benchmark tables into engine-ready planet
specs + ground-truth 12-vectors.

Consumes:
  benchmarks/solar_system_truth.json  — literal ground truth (Section E)
  benchmarks/published_retrievals.json — literature pseudo-truth (Section H)

Each returned target carries:
  planet       — dict in the engines' expected schema (star/planet params +
                 12-species VMR ``composition`` floored to LOG_FLOOR)
  true_vector  — np.ndarray[12] linear VMR in TARGET_COLUMNS order
  metadata     — name, provenance, and (E) representable / (H) observable+domain_match.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_THIS, os.pardir))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from common.data import TARGET_COLUMNS   # noqa: E402

BENCH_DIR = os.path.join(_THIS, "benchmarks")
LOG_FLOOR = 1e-12


def _load(name):
    with open(os.path.join(BENCH_DIR, name)) as f:
        return json.load(f)


def _target_vector(vmr):
    """12-vec in TARGET_COLUMNS order, absent species floored (never exact zero)."""
    return np.array([max(float(vmr.get(s, 0.0)), LOG_FLOOR) for s in TARGET_COLUMNS], dtype=float)


def _composition(vmr):
    """Full 12-species VMR dict for the engine (floored)."""
    return {s: max(float(vmr.get(s, 0.0)), LOG_FLOOR) for s in TARGET_COLUMNS}


def _planet(params, vmr, star):
    p = dict(params)
    p.setdefault("star_temperature", star["star_temperature"])
    p.setdefault("star_radius", star["star_radius"])
    p.setdefault("star_mass", star.get("star_mass", 1.0))
    p["composition"] = _composition(vmr)
    return p


def solar_system_targets():
    """Section E: Solar-System bodies as exoplanets with known composition."""
    doc = _load("solar_system_truth.json")
    star = doc["star_sun"]
    out = []
    for b in doc["bodies"]:
        out.append({
            "name": b["name"],
            "planet": _planet(b["params"], b["vmr"], star),
            "true_vector": _target_vector(b["vmr"]),
            "representable": bool(b.get("representable", True)),
            "covered_species": b.get("covered_species", []),
            "note": b.get("note", ""),
            "source": "; ".join(doc["_sources"][:1]),
        })
    return out


def published_targets():
    """Section H: benchmark exoplanets with literature-retrieved pseudo-truth."""
    doc = _load("published_retrievals.json")
    # Published targets are giants around their own stars; a Sun default only fills
    # any missing star field (all entries specify star_temperature/_radius).
    star = {"star_temperature": 5772.0, "star_radius": 1.0, "star_mass": 1.0}
    out = []
    for t in doc["targets"]:
        vmr = {k: 10.0 ** v for k, v in t["retrieved_log10_vmr"].items()}
        out.append({
            "name": t["name"],
            "planet": _planet(t["params"], vmr, star),
            "true_vector": _target_vector(vmr),
            "retrieved_species": list(t["retrieved_log10_vmr"].keys()),
            "observable": t.get("observable", "unknown"),
            "domain_match": t.get("domain_match", "far"),
            "citation": t.get("citation", ""),
            "doi": t.get("doi", ""),
            "note": t.get("note", ""),
        })
    return out
