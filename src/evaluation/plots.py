"""Shared plotting helpers for the evaluation suites (matplotlib Agg).

Every function returns the path it wrote. Suites call these inside try/except so a
plotting failure (e.g. a headless quirk) never breaks the numeric evaluation.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def sweep_plot(out_dir, label, sweep, ref_r2=None, fname="r2_vs_snr.png"):
    xs = [r["snr_planet_band"] for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [r["r2_covered"] for r in sweep], "o-", color="navy", label="covered species")
    ax.plot(xs, [r["r2_all12"] for r in sweep], "s--", color="gray", alpha=0.6, label="all 12")
    if ref_r2 is not None:
        ax.axhline(ref_r2, color="darkorange", ls="--", alpha=0.8,
                   label=f"noiseless ref ({ref_r2:.3f})")
    ax.axvline(5.0, color="crimson", ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Band-integrated planet detection SNR (median)")
    ax.set_ylabel("R² (log10)")
    ax.set_title(f"{label}: retrieval vs SNR / exposure")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def truth_bars(out_dir, name, species, true_vmr, pred_vmr, covered=None, fname=None):
    """Known-truth bar chart: true vs predicted VMR (log), covered species highlighted."""
    x = np.arange(len(species))
    w = 0.38
    covered = set(covered or species)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w / 2, np.clip(true_vmr, 1e-12, None), w, label="truth", color="tab:green")
    colors = ["tab:blue" if s in covered else "lightgray" for s in species]
    ax.bar(x + w / 2, np.clip(pred_vmr, 1e-12, None), w, label="predicted", color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45)
    ax.set_ylabel("mole fraction (log)")
    ax.set_title(f"{name}: known composition vs predicted (grey = uncovered)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, fname or f"truth_bars_{name}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def reliability_plot(out_dir, label, rel, fname="reliability.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="ideal")
    ax.plot(rel["nominal"], rel["empirical_mean"], "o-", color="navy",
            label=f"empirical (ECE={rel['ece']:.3f})")
    ax.set_xlabel("nominal central coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"{label}: calibration reliability")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def sbc_hist(out_dir, label, ranks, n_samples, species, fname="sbc_ranks.png"):
    D = ranks.shape[1]
    ncol = 4
    nrow = int(np.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        if i >= D:
            ax.axis("off")
            continue
        ax.hist(ranks[:, i], bins=20, range=(0, n_samples + 1), color="slateblue", alpha=0.8)
        ax.axhline(ranks.shape[0] / 20, color="crimson", ls="--", alpha=0.6)
        ax.set_title(species[i], fontsize=9)
        ax.set_xticks([])
    fig.suptitle(f"{label}: SBC rank histograms (flat = calibrated)")
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def tarp_plot(out_dir, label, tarp_res, fname="tarp_coverage.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="ideal")
    ax.plot(tarp_res["alpha"], tarp_res["coverage"], "o-", color="darkgreen",
            label=f"TARP (ECE={tarp_res['ece_tarp']:.3f})")
    ax.set_xlabel("credibility level")
    ax.set_ylabel("expected coverage")
    ax.set_title(f"{label}: TARP expected coverage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def delta_lambda_plot(out_dir, label, wl, delta, vratio, fname="ood_delta_lambda.png"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(wl, delta, color="firebrick", lw=0.7)
    a1.axhspan(-0.5, 0.5, color="green", alpha=0.1)
    a1.set_ylabel("δ(λ)  (std offset)")
    a1.set_title(f"{label}: OOD shift vs INARA after the INARA-fit norm")
    a2.plot(wl, vratio, color="navy", lw=0.7)
    a2.axhspan(0.5, 2.0, color="green", alpha=0.1)
    a2.set_ylabel("variance ratio v(λ)")
    a2.set_xlabel("wavelength (µm)")
    a2.set_yscale("log")
    fig.tight_layout()
    p = os.path.join(out_dir, fname)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
