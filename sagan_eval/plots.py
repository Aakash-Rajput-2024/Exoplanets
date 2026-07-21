"""Three figures for the Sagan-catalog evaluation.  PYTHONPATH=src:. python3 sagan_eval/plots.py"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))
HERE = os.path.dirname(os.path.abspath(__file__))

from sagan_eval.analyze import analyze_track, inara_o2o3_prior   # noqa: E402

CLR = {"terrestrial": "#2c7fb8", "giant": "#d95f0e", "airless": "#666666"}


def fig_o2(doc, track="causal", out="fig_o2_fabrication.png"):
    rows = doc["results"][track]["clear"]
    rows = sorted(rows, key=lambda r: -r["pred"]["O2"])
    names = [r["body"] for r in rows]
    vals = [r["pred"]["O2"] for r in rows]
    cols = [CLR[r["body_class"]] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(vals)), vals, color=cols)
    ax.axhline(0.328, ls="--", c="k", lw=1, label="model's mean output on INARA spectra (0.328)")
    ax.axhline(0.2095, ls=":", c="#2c7fb8", lw=1.6, label="Earth's TRUE O$_2$ (0.2095)")
    ax.axhline(0.0, c="k", lw=0.8)
    ie = names.index("Earth")
    ax.annotate("Earth\n(the only body\nwith any O$_2$)", (ie, vals[ie]),
                textcoords="offset points", xytext=(0, 16), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("predicted O$_2$ VMR")
    ax.set_title(f"{track}: predicted O$_2$ on 19 real solar-system spectra — "
                 "airless bodies (grey) have exactly zero O$_2$")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLR.values()]
    ax.legend(h + ax.get_legend_handles_labels()[0],
              list(CLR) + ax.get_legend_handles_labels()[1], fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, out), dpi=150); plt.close(fig)
    return out


def fig_auroc(doc, out="fig_ch4_auroc.png"):
    tracks = list(doc["results"])
    clear = [analyze_track(doc["results"][t]["clear"])["auroc_CH4_hires"] for t in tracks]
    dec = [analyze_track(doc["results"][t]["declouded"])["auroc_CH4_hires"]
           if "declouded" in doc["results"][t] else np.nan for t in tracks]
    x = np.arange(len(tracks)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, clear, w, label="clear (as delivered)", color="#2c7fb8")
    ax.bar(x + w / 2, dec, w, label="after declouding", color="#d95f0e")
    ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(tracks, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CH$_4$ detection AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Matched-resolution CH$_4$ detection (5 CH$_4$-rich vs 5 airless, all R=829-867)\n"
                 "declouding destroys the only real signal", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, out), dpi=150); plt.close(fig)
    return out


def fig_earth_moon(doc, track="causal", out="fig_earth_moon.png"):
    from common.data import TARGET_COLUMNS
    rows = {r["body"]: r for r in doc["results"][track]["clear"]}
    e, m = rows["Earth"], rows["Moon"]
    x = np.arange(len(TARGET_COLUMNS)); w = 0.38
    true_e = {"N2": 0.7808, "O2": 0.2095, "H2O": 0.01, "CO2": 4.2e-4}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, [e["pred"][c] for c in TARGET_COLUMNS], w, label="Earth (pred)", color="#2c7fb8")
    ax.bar(x + w / 2, [m["pred"][c] for c in TARGET_COLUMNS], w, label="Moon (pred)", color="#666666")
    ax.plot(x - w / 2, [true_e.get(c, 0) for c in TARGET_COLUMNS], "k*", ms=9, ls="none",
            label="Earth TRUE")
    ax.plot(x + w / 2, np.zeros(len(x)), "rx", ms=6, ls="none", label="Moon TRUE (all zero)")
    ax.set_xticks(x); ax.set_xticklabels(TARGET_COLUMNS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("VMR")
    ax.set_title(f"{track}: Earth vs Moon — same night (2008-11-21), same instrument, both R=867",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, out), dpi=150); plt.close(fig)
    return out


if __name__ == "__main__":
    with open(os.path.join(HERE, "sagan_results.json")) as f:
        doc = json.load(f)
    for fn in (fig_o2, fig_auroc, fig_earth_moon):
        print("wrote", fn(doc))
