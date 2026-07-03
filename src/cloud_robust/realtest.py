"""Run the original vs cloud-robust models on a REAL Earth-as-an-exoplanet spectrum.

Spectrum: the Robinson et al. (2011) VPL 3-D Earth model at quadrature
(`data/real_earth/earth_quadrature_radiance_refl.dat`) — disk-integrated Earth
reflected light, validated against EPOXI + earthshine, with Earth's REAL ~60%
cloud cover baked in. That last point makes it the ideal real test of this
project: Earth is a cloudy reflected-light planet, so a cloud-robust model
should read its composition more sensibly than the cloud-blind original.

This is a SINGLE-target QUALITATIVE test (one planet, no R^2), with important
caveats printed in the report:
  - observable/units differ from INARA (we use the reflected-intensity column,
    regridded onto INARA's grid and magnitude-matched to INARA's level before
    standardizing with each model's own training stats);
  - INARA's training LABEL distribution samples thick, exotic atmospheres, so
    Earth's ppm/ppb trace gases (CH4, O3, ...) are out-of-distribution — expect
    the legible signal to be in the MAJOR gases (N2, O2, H2O, CO2).

ISOLATION: reads checkpoints/caches/spectra read-only; writes only under
src/cloud_robust/realtest_out/.

    python realtest.py                 # default: transformerarch (1-channel)
    python realtest.py original1dcnn   # any 1-channel track
"""

from __future__ import annotations

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from eval_cloud_robustness import EVAL, _load_model, _load_stats, _device, TARGETS  # noqa: E402

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
DATA = os.path.join(REPO, "data")
EARTH_DAT = os.path.join(DATA, "real_earth", "earth_quadrature_radiance_refl.dat")
INARA_DIR = os.path.join(DATA, "inara_1by3")

# Earth's approximate atmospheric composition (volume mixing ratios). Trace-gas
# values are order-of-magnitude and vary; shown for context, not as exact truth.
EARTH_VMR = {
    "H2O": 1e-2, "CO2": 4.2e-4, "O2": 0.209, "N2": 0.781, "CH4": 1.9e-6,
    "N2O": 3.3e-7, "CO": 1e-7, "O3": 7e-7, "SO2": 1e-10, "NH3": 1e-9,
    "C2H6": 1e-9, "NO2": 1e-11,
}

# Single-channel tracks only (Earth reflected intensity = the planet_signal analog).
SUPPORTED = ["original1dcnn", "transformerarch", "causal"]


def load_inara_grid():
    """INARA wavelength grid (<=2.0 um) from one real INARA spectrum."""
    f = sorted(glob.glob(os.path.join(INARA_DIR, "*.csv")))[0]
    df = pd.read_csv(f)
    wl = df.loc[df["wavelength_(um)"] <= 2.0, "wavelength_(um)"].to_numpy()
    return wl


def load_earth():
    """Return (wavelength_um, reflected_intensity) from the Robinson Earth file."""
    rows = []
    for ln in open(EARTH_DAT):
        ln = ln.strip()
        if not ln or ln.startswith(";"):
            continue
        try:
            rows.append([float(x) for x in ln.split()])
        except ValueError:
            pass
    a = np.array(rows)
    return a[:, 0], a[:, 1]  # wavelength [um], intensity [W/m^2/um/sr]


def standardize_for_model(earth_on_grid, mean_x, std_x):
    """Magnitude-match Earth to INARA's level (so it lands in-distribution), then
    standardize channel-wise with the model's training stats.

    mean_x/std_x are (1, C, 1). Earth is single-channel reflected light, so we use
    channel-0 stats. Calibration: scale Earth so its mean equals INARA's raw mean
    (mean_x); the informative band structure is preserved under global scaling.
    """
    mx = float(mean_x.flatten()[0])
    sx = float(std_x.flatten()[0])
    e = earth_on_grid.astype(np.float64)
    scale = mx / (e.mean() + 1e-300)
    e_scaled = e * scale
    e_std = (e_scaled - mx) / (sx + 1e-30)
    return torch.tensor(e_std, dtype=torch.float32).view(1, 1, -1)


def predict_abundances(model, x_std, mean_y, std_y, device):
    model.eval()
    with torch.no_grad():
        pred = model(x_std.to(device)).cpu()
    return (pred * std_y + mean_y).numpy().flatten()


def run(track="transformerarch"):
    if track not in SUPPORTED:
        raise SystemExit(f"realtest supports 1-channel tracks {SUPPORTED}; got '{track}'.")
    cfg = EVAL[track]
    device = _device()
    out_dir = os.path.join(THIS_DIR, "realtest_out")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Real Earth test: {track} on {device} ===")

    grid = load_inara_grid()
    ewl, eint = load_earth()
    earth_on_grid = np.interp(grid, ewl, eint)  # Earth fully covers 0.2-2um
    print(f"  INARA grid {len(grid)} pts ({grid.min():.3f}-{grid.max():.3f} um); "
          f"Earth regridded (raw intensity {eint.min():.2e}..{eint.max():.2e})")

    seq_len = len(grid)
    s_orig = _load_stats(cfg["orig_stats"])
    s_rob = _load_stats(cfg["robust_stats"])
    m_orig, ep_orig = _load_model(cfg["model_py"], cfg["cls"], cfg["orig_ckpt"], 1, seq_len, device)
    m_rob, ep_rob = _load_model(cfg["model_py"], cfg["cls"], cfg["robust_ckpt"], 1, seq_len, device)

    preds = {}
    if m_orig is not None:
        x = standardize_for_model(earth_on_grid, s_orig[0], s_orig[1])
        preds["original"] = predict_abundances(m_orig, x, s_orig[2], s_orig[3], device)
    else:
        print(f"  [skip] original checkpoint missing: {cfg['orig_ckpt']}")
    if m_rob is not None:
        x = standardize_for_model(earth_on_grid, s_rob[0], s_rob[1])
        preds["robust"] = predict_abundances(m_rob, x, s_rob[2], s_rob[3], device)
    else:
        print(f"  [warn] robust checkpoint missing: {cfg['robust_ckpt']} (train it first)")

    _report(track, out_dir, preds, {"original": ep_orig, "robust": ep_rob})
    _plot(track, out_dir, grid, earth_on_grid, preds)
    print(f"=> Wrote details.txt + charts to {out_dir}")
    return preds


def _report(track, out_dir, preds, epochs):
    L = []
    L.append("=" * 78)
    L.append(f"REAL EARTH-AS-EXOPLANET TEST — {track}")
    L.append("Spectrum: Robinson et al. 2011 VPL 3-D Earth model (quadrature, real clouds)")
    L.append("=" * 78)
    eo, er = epochs.get("original"), epochs.get("robust")
    L.append(f"Num epochs (best checkpoint):  original={eo if eo is not None else 'n/a'}   "
             f"cloud-robust={er if er is not None else 'n/a'}")
    L.append("Single target (Earth) -> qualitative read, no R^2.")
    L.append("")
    L.append("Predicted abundance (mole fraction) vs Earth's approx VMR:")
    L.append(f"  {'species':<8} | {'Earth VMR':>10} | {'original':>10} | {'cloud-robust':>12}")
    L.append("-" * 60)
    for i, sp in enumerate(TARGETS):
        ev = EARTH_VMR[sp]
        o = preds.get("original", [np.nan] * 12)[i]
        r = preds.get("robust", [np.nan] * 12)[i]
        L.append(f"  {sp:<8} | {ev:>10.2e} | {o:>10.4f} | {r:>12.4f}")
    L.append("-" * 60)
    # Major-gas legibility: which model ranks N2/O2/H2O/CO2 closer to Earth?
    major = ["N2", "O2", "H2O", "CO2"]
    L.append("")
    L.append("Major-gas check (the legible part; trace gases are out of INARA's range):")
    for sp in major:
        i = TARGETS.index(sp)
        ev = EARTH_VMR[sp]
        o = preds.get("original", [np.nan] * 12)[i]
        r = preds.get("robust", [np.nan] * 12)[i]
        do = abs(o - ev); dr = abs(r - ev)
        better = "robust" if dr < do else "original"
        L.append(f"  {sp:<4} Earth={ev:6.3f}  orig={o:7.4f} (|err| {do:.3f})  "
                 f"robust={r:7.4f} (|err| {dr:.3f})  -> closer: {better}")
    L.append("")
    L.append("CAVEATS: (1) units/observable adapted from W/m^2/um/sr reflected intensity, "
             "magnitude-matched to INARA before standardizing. (2) INARA labels sample thick "
             "exotic atmospheres, so Earth's ppm/ppb trace gases are OUT of distribution; both "
             "models will over-predict them. (3) One planet, so this is indicative, not statistical.")
    L.append("=" * 78)
    report = "\n".join(L)
    print(report)
    with open(os.path.join(out_dir, f"realtest_{track}_details.txt"), "w") as fh:
        fh.write(report + "\n")


def _plot(track, out_dir, grid, earth_on_grid, preds):
    charts = os.path.join(out_dir, "charts")
    os.makedirs(charts, exist_ok=True)
    # (1) Earth spectrum on the INARA grid
    plt.figure(figsize=(11, 5))
    plt.plot(grid, earth_on_grid, lw=0.7, color="tab:green")
    plt.xlabel("wavelength (um)"); plt.ylabel("reflected intensity [W/m^2/um/sr]")
    plt.title(f"Real Earth (Robinson 2011) regridded onto INARA grid — {track}")
    plt.tight_layout()
    plt.savefig(os.path.join(charts, f"earth_spectrum_{track}.png"), dpi=150)
    plt.close()

    # (2) predicted abundances: Earth vs original vs robust (log scale)
    x = np.arange(len(TARGETS)); w = 0.27
    ev = np.array([EARTH_VMR[s] for s in TARGETS])
    plt.figure(figsize=(14, 6))
    plt.bar(x - w, np.clip(ev, 1e-12, None), w, label="Earth VMR", color="tab:green")
    if "original" in preds:
        plt.bar(x, np.clip(preds["original"], 1e-12, None), w, label="original", color="tab:red", alpha=0.8)
    if "robust" in preds:
        plt.bar(x + w, np.clip(preds["robust"], 1e-12, None), w, label="cloud-robust", color="tab:blue", alpha=0.8)
    plt.yscale("log")
    plt.xticks(x, TARGETS, rotation=45); plt.ylabel("mole fraction (log)")
    plt.title(f"Earth composition: truth vs predicted — {track}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(charts, f"earth_abundances_{track}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "transformerarch")
