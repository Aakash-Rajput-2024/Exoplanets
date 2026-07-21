"""Generate ALL report figures in black-and-white, well-named, into report/figures/.

Grayscale only: fills distinguished by shade + hatch; lines by style + marker.
Run:  PYTHONPATH=src:. python3 sagan_eval/make_report_figs.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT = os.path.join(REPO, "report", "figures")
os.makedirs(OUT, exist_ok=True)
SAGAN = os.path.join(REPO, "sagan_eval")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10, "axes.edgecolor": "black",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "0.8",
    "grid.linewidth": 0.5, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})
GREYS = ["0.15", "0.45", "0.70", "0.88"]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "ooo"]


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, REPO))


# ---- 1. model accuracy + stack ------------------------------------------------
def fig_model_accuracy():
    models = ["causal", "original1dcnn", "optimized1dcnn", "transformerarch",
              "causal_cfi", "causal_xl"]
    r2 = [0.781, 0.779, 0.728, 0.568, 0.355, 0.173]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(models))[::-1]
    ax.barh(y, r2, color="0.55", edgecolor="black", height=0.6)
    ax.barh([len(models)], [0.849], color="white", edgecolor="black", hatch="///", height=0.6)
    ax.set_yticks(list(y) + [len(models)])
    ax.set_yticklabels(models + ["STACK (all 6)"])
    for yi, v in zip(list(y) + [len(models)], r2 + [0.849]):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("$R^2$ (covered species, log$_{10}$, noiseless)")
    ax.set_xlim(0, 1.0)
    ax.set_title("Model accuracy — best single model is also the smallest (520k)")
    save(fig, "fig01_model_accuracy.png")


# ---- 2. cross-generator adaptation -------------------------------------------
def fig_crossgen():
    cond = ["PSG\n(native)", "pRT\n(raw)", "pRT\n+AdaBN", "pRT\n+affine"]
    vals = [0.728, -3.736, -0.162, -0.038]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(cond))
    bars = ax.bar(x, vals, color=["0.3", "0.85", "0.6", "0.4"],
                  edgecolor="black", width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.1 if v >= 0 else -0.35),
                f"{v:+.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cond)
    ax.set_ylabel("$R^2$ covered")
    ax.set_title("Cross-generator: catastrophic gap is mostly removable (optimized1dcnn)")
    save(fig, "fig02_crossgen_adaptation.png")


# ---- 3. per-gas R2 noiseless vs alpha=1 --------------------------------------
def fig_pergas():
    gases = ["H2O", "CO2", "O2", "CH4", "O3", "N2"]
    nl = [0.72, 0.73, 0.80, 0.79, 0.86, -0.06]
    a1 = [0.06, 0.03, 0.12, 0.11, 0.33, -0.06]
    x = np.arange(len(gases)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - w/2, nl, w, color="0.35", edgecolor="black", label="noiseless (best case)")
    ax.bar(x + w/2, a1, w, color="white", edgecolor="black", hatch="///",
           label="$\\alpha$=1 (one real exposure)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(gases)
    ax.set_ylabel("$R^2$ (log$_{10}$)"); ax.legend(fontsize=8)
    ax.set_title("Per-gas accuracy: strong at high signal, noise-limited at $\\alpha$=1")
    save(fig, "fig03_pergas_accuracy.png")


# ---- 4. exposure vs accuracy -------------------------------------------------
def fig_exposure():
    tracks = {"original1dcnn": "-o", "optimized1dcnn": "--s", "transformerarch": ":^",
              "causal": "-D", "causal_cfi_cfi": "--v", "causal_xl_cfi": ":P"}
    shades = ["0.0", "0.25", "0.45", "0.1", "0.55", "0.65"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for (t, sty), sh in zip(tracks.items(), shades):
        p = os.path.join(REPO, f"src/evaluation/results/{t}/in_distribution/result.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        tab = next(x for x in d["tables"] if "SNR sweep" in x["name"])
        exp = [r[1] for r in tab["rows"]]; r2 = [r[3] for r in tab["rows"]]
        ax.plot(exp, r2, sty, color=sh, ms=5, lw=1.3,
                label=t.replace("_cfi", "").replace("_", " "), markerfacecolor="white")
    ax.set_xscale("log"); ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("exposure time  $\\times$ nominal (8 hr)")
    ax.set_ylabel("$R^2$ covered")
    ax.set_title("Accuracy is photon-limited: more exposure, more accuracy")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "fig04_exposure_accuracy.png")


# ---- 5. cloud generalization -------------------------------------------------
def fig_cloud():
    fam = ["grey\n(seen)", "non-grey\nhaze", "band-\nselective", "patchy\n(Earth)"]
    cloudy = [-1.02, -0.51, -0.61, -0.97]
    decl = [-0.19, -0.19, -0.28, -0.21]
    x = np.arange(len(fam)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w/2, cloudy, w, color="0.8", edgecolor="black", label="clouded")
    ax.bar(x + w/2, decl, w, color="0.35", edgecolor="black", label="de-clouded")
    ax.axhline(0.788, color="black", lw=1, ls="--", label="clear-sky (0.79)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(fam)
    ax.set_ylabel("$R^2$ covered"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("De-clouding recovers 24–46% of cloud loss, incl. unseen cloud types")
    save(fig, "fig05_cloud_generalization.png")


# ---- 6. conformal coverage ---------------------------------------------------
def fig_conformal():
    gases = ["H2O", "CO2", "O2", "CH4", "O3"]
    nat = [0.71, 0.71, 0.67, 0.67, 0.67]
    r22 = [0.36, 0.20, 0.26, 0.41, 0.16]
    x = np.arange(len(gases)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x - w/2, nat, w, color="0.35", edgecolor="black", label="native resolution")
    ax.bar(x + w/2, r22, w, color="white", edgecolor="black", hatch="xxx", label="blurred (R=22)")
    ax.axhline(0.68, color="black", lw=1, ls="--", label="target 0.68")
    ax.set_xticks(x); ax.set_xticklabels(gases)
    ax.set_ylabel("interval coverage"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Error bars are honest natively, too narrow when data is blurred")
    save(fig, "fig06_uncertainty_coverage.png")


# ---- 7. CH4 AUROC ------------------------------------------------------------
def fig_ch4():
    doc = json.load(open(os.path.join(SAGAN, "sagan_results.json")))
    from sagan_eval.analyze import analyze_track
    tracks = list(doc["results"])
    vals = [analyze_track(doc["results"][t]["clear"])["auroc_CH4_hires"] for t in tracks]
    order = np.argsort(vals)[::-1]
    tracks = [tracks[i] for i in order]; vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(tracks))
    ax.bar(x, vals, color="0.4", edgecolor="black", width=0.6)
    ax.axhline(0.5, color="black", lw=1, ls="--", label="chance (0.5)")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_cfi", "").replace("_", " ") for t in tracks],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CH$_4$ detection AUROC"); ax.set_ylim(0, 1.08); ax.legend(fontsize=8)
    ax.set_title("Methane detection on real data (5 CH$_4$-rich vs 5 airless, matched R)")
    save(fig, "fig07_ch4_detection_auroc.png")


# ---- 8. O2 fabrication (19 bodies) -------------------------------------------
def fig_o2():
    doc = json.load(open(os.path.join(SAGAN, "sagan_results.json")))
    rows = sorted(doc["results"]["causal"]["clear"], key=lambda r: -r["pred"]["O2"])
    names = [r["body"] for r in rows]; vals = [r["pred"]["O2"] for r in rows]
    shade = {"terrestrial": "0.15", "giant": "0.5", "airless": "0.8"}
    cols = [shade[r["body_class"]] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="black")
    ax.axhline(0.328, color="black", ls="--", lw=1, label="model's average guess (0.328)")
    ax.axhline(0.2095, color="black", ls=":", lw=1.4, label="Earth's TRUE O$_2$ (0.21)")
    ie = names.index("Earth")
    ax.annotate("Earth", (ie, vals[ie]), textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("predicted O$_2$")
    hs = [plt.Rectangle((0, 0), 1, 1, fc=shade[k], ec="black") for k in shade]
    ax.legend(hs + ax.get_legend_handles_labels()[0],
              ["terrestrial", "giant", "airless (no atmosphere)"]
              + ax.get_legend_handles_labels()[1], fontsize=7, ncol=2)
    ax.set_title("Predicted O$_2$ on 19 real bodies — bare rocks out-score Earth")
    save(fig, "fig08_o2_fabrication.png")


# ---- 9. Earth vs Moon --------------------------------------------------------
def fig_earth_moon():
    doc = json.load(open(os.path.join(SAGAN, "sagan_results.json")))
    tracks = list(doc["results"])
    e = [next(r for r in doc["results"][t]["clear"] if r["body"] == "Earth")["pred"]["O2"] for t in tracks]
    m = [next(r for r in doc["results"][t]["clear"] if r["body"] == "Moon")["pred"]["O2"] for t in tracks]
    x = np.arange(len(tracks)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - w/2, e, w, color="0.3", edgecolor="black", label="Earth (true 0.21)")
    ax.bar(x + w/2, m, w, color="white", edgecolor="black", hatch="///", label="Moon (true 0.00)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_cfi", "").replace("_", " ") for t in tracks],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("predicted O$_2$"); ax.legend(fontsize=8)
    ax.set_title("Earth vs Moon (same night, same instrument): rock $\\geq$ Earth in O$_2$")
    save(fig, "fig09_earth_vs_moon.png")


# ---- 10. resolution curve ----------------------------------------------------
def fig_resolution():
    xr = [22, 50, 100, 200, 400, 870, 1900]
    raw = {  # verified INARA-test values (causal), raw R2 vs resolution
        "H2O": [-0.24, 0.48, 0.55, 0.64, 0.60, 0.59, 0.71],
        "CO2": [-1.87, -4.37, -1.82, 0.39, 0.68, 0.70, 0.74],
        "O2":  [-0.15, 0.26, 0.53, 0.71, 0.78, 0.80, 0.80],
        "CH4": [0.32, 0.50, 0.64, 0.74, 0.79, 0.80, 0.78],
        "O3":  [0.27, 0.55, 0.67, 0.76, 0.81, 0.83, 0.83],
    }
    sty = ["-o", "--s", ":^", "-.D", "-v"]
    sh = ["0.0", "0.0", "0.35", "0.35", "0.6"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, s, c in zip(raw, sty, sh):
        ax.plot(xr, raw[g], s, color=c, ms=5, lw=1.3, label=g, markerfacecolor="white")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(200, color="black", lw=0.9, ls="--")
    ax.text(210, -1.6, "CO$_2$ floor\nR$\\geq$200", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("spectral resolution R")
    ax.set_ylabel("$R^2$ (log$_{10}$)"); ax.set_ylim(-2, 1)
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_title("Which gas needs how sharp a spectrum (CO$_2$ is the demanding one)")
    save(fig, "fig10_resolution_requirements.png")


# ---- 11. twin nulling --------------------------------------------------------
def fig_twin():
    labels = ["O2", "CO2", "N2"]
    raw = [0.341, 0.351, 0.255]
    nulled = [0.016, 0.013, 0.003]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x - w/2, raw, w, color="0.8", edgecolor="black", label="raw (fabricated)")
    ax.bar(x + w/2, nulled, w, color="0.2", edgecolor="black", label="after twin-nulling")
    for xi, v in zip(x - w/2, raw):
        ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, nulled):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"|{g}| on rocks" for g in labels])
    ax.set_ylabel("mean predicted abundance"); ax.legend(fontsize=8)
    ax.set_title("Twin-nulling removes fabricated gas on airless bodies (causal)")
    save(fig, "fig11_twin_nulling.png")


# ---- 12. stacking ------------------------------------------------------------
def fig_stack():
    labels = ["best single", "plain average", "per-gas STACK"]
    vals = [0.777, 0.637, 0.849]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(range(3), vals, color=["0.55", "0.8", "0.2"], edgecolor="black", width=0.6)
    bars[2].set_hatch("///")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("$R^2$ covered"); ax.set_ylim(0, 0.95)
    ax.set_title("Combining the 6 frozen models beats the best single (no retraining)")
    save(fig, "fig12_model_stacking.png")


if __name__ == "__main__":
    fig_model_accuracy(); fig_crossgen(); fig_pergas(); fig_exposure()
    fig_cloud(); fig_conformal(); fig_ch4(); fig_o2(); fig_earth_moon()
    fig_resolution(); fig_twin(); fig_stack()
    print("\nAll figures in report/figures/")
