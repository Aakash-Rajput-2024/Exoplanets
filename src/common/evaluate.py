"""Shared test-set evaluator + SNR sweep (fixes C3, reports C2, delivers C1 figure).

Every track is scored the same way, on the frozen TEST split it never selected on
(C3), in log₁₀ mixing-ratio space with honest R² and bootstrap CIs (C2), across
all trained seeds (H6). Predictions are decoded through the CLR pipeline, so they
are always valid simplices — no negative abundances (C2).

Two products per track, written to ``<track_dir>/eval_v2/``:
  1. ``details_v2.txt`` — per-species R²/RMSE/MAE (dex) with 95% CIs, seed-ensemble
     point estimate + per-seed overall R² mean±std.
  2. SNR sweep — overall R² vs exposure time / achieved SNR. α = √(t/t_nominal)
     is PHYSICAL (noise units validated: LUVOIR-like 8 hr @ 5 pc nominal,
     Zorzan+25 §3.1): α=1 is the photon-starved floor (R²≈0 expected), the sweep
     crosses the planet-detection threshold (band SNR≈5) around α~50-100. This is
     the C1 headline figure. CSV + PNG (planet-SNR bottom axis, exposure top).
     Also reports a NOISELESS CEILING (zero injected contrast noise, i.e. the
     true α→∞ limit) alongside the sweep, so "is the accuracy good?" has a
     concrete answer: the % of that ceiling reached at each realistic exposure.
     A sweep that's already near the ceiling means more exposure won't help —
     the observable/architecture, not photon noise, is the limit.

Usage:
    python -m common.evaluate <track> [--seeds 0 1 2] [--alphas 0.3 1 3 10 30 100 300]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.registry import track_config, load_model_class, CACHE_V2
from common.runtime import get_device
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED
from common.inputs import build_eval_observable
from common.observable import snr_summary, make_observable
from common import metrics

TARGET_COLUMNS = ["H2O", "CO2", "O2", "N2", "CH4", "N2O",
                  "CO", "O3", "SO2", "NH3", "C2H6", "NO2"]
DEFAULT_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
CEILING_ALPHA = 1e4  # reporting-only label for the noiseless row; see evaluate_track


def _ckpt_path(track, seed, suffix=""):
    track_dir = os.path.dirname(track_config(track)["model_py"])
    return os.path.join(track_dir, "checkpoints_v2", f"model_best_seed{seed}{suffix}.pth")


def _load_model(track, seed, device, suffix=""):
    cfg = track_config(track)
    # weights_only=False: our checkpoints intentionally carry provenance metadata
    # (config, per-epoch history — the C6 stamp), not just tensors. They are our
    # own trusted local files, so the full unpickler is the correct choice; the
    # PyTorch-2.6 weights_only=True default rejects the embedded numpy/py objects.
    ck = torch.load(_ckpt_path(track, seed, suffix), map_location=device, weights_only=False)
    ModelClass = load_model_class(cfg["model_py"], cfg["model_cls"])
    model = ModelClass(in_channels=ck["in_channels"], sequence_length=ck["seq_len"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("config", {})


def _ckpt_provenance(track, seed, device, suffix=""):
    """Read the ACTUAL run parameters stamped into a checkpoint (not the
    aspirational MATCHED budget), so reports state what really ran (C6/§2.3-4)."""
    ck = torch.load(_ckpt_path(track, seed, suffix), map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    return {
        "epochs_run": len(ck.get("history", [])),
        "epochs_planned": c.get("epochs"),
        "best_epoch": ck.get("epoch"),
        "batch_size": c.get("batch_size"),
        "loss": c.get("loss"),
        "lr": c.get("lr"),
        "obs_mode": c.get("obs_mode"),
        "label_transform": c.get("label_transform"),
        "input_norm": c.get("input_norm"),
        "git_commit": ck.get("git_commit"),
        "config_hash": ck.get("config_hash"),
    }


@torch.no_grad()
def _predict(model, x_obs, device, batch=256):
    out = []
    for i in range(0, x_obs.shape[0], batch):
        out.append(model(x_obs[i:i + batch].to(device)).cpu())
    return torch.cat(out)


def evaluate_track(track, seeds=None, cache_dir=CACHE_V2, alphas=DEFAULT_ALPHAS):
    cfg = track_config(track)
    seeds = list(seeds) if seeds is not None else list(cfg["seeds"])
    seeds = [s for s in seeds if os.path.exists(_ckpt_path(track, s))]
    if not seeds:
        raise SystemExit(f"No checkpoints found for {track} (looked for model_best_seed*.pth)")
    device = get_device()
    out_dir = os.path.join(os.path.dirname(cfg["model_py"]), "eval_v2")
    os.makedirs(out_dir, exist_ok=True)

    raw_x, y_lin, noise, _ids, lp = load_eval_raw(cache_dir, "test", cfg["label_transform"])
    y_true = y_lin.numpy()
    # obs/norm config read from a checkpoint (config-driven, robust to flag changes)
    _, ck_cfg = _load_model(track, seeds[0], device)
    obs_mode = ck_cfg.get("obs_mode", cfg["obs_mode"])
    input_norm = ck_cfg.get("input_norm", cfg["input_norm"])
    norm = get_norm(cache_dir, obs_mode, input_norm)

    # --- headline: α=1 test observable, per-seed + ensemble --------------------
    x1 = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=1.0, seed=TEST_NOISE_SEED)
    per_seed_r2, seed_preds = [], []
    for s in seeds:
        model, _ = _load_model(track, s, device)
        pred_lin = lp.decode(_predict(model, x1, device)).numpy()
        seed_preds.append(pred_lin)
        rep = metrics.report(y_true, pred_lin, space="log10", bootstrap=0)
        per_seed_r2.append(float(np.nanmean(rep["r2"])))
    ens = np.mean(seed_preds, axis=0)                     # seed-ensemble prediction
    rep = metrics.report(y_true, ens, space="log10", bootstrap=1000)

    # --- SNR sweep (α = √(t/t_nom); the C1 headline figure) --------------------
    sweep = []
    for a in alphas:
        xa = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=a, seed=TEST_NOISE_SEED)
        r2s = []
        for s in seeds:
            model, _ = _load_model(track, s, device)
            pred = lp.decode(_predict(model, xa, device)).numpy()
            r2s.append(float(np.nanmean(metrics.report(y_true, pred, space="log10", bootstrap=0)["r2"])))
        row = dict(alpha=a, **snr_summary(raw_x[:, 0], raw_x[:, 1], noise, a))
        row.update(r2_log10_mean=float(np.mean(r2s)), r2_log10_std=float(np.std(r2s)))
        sweep.append(row)

    # --- Noiseless ceiling: how much of the observable's information is being
    # extracted, independent of exposure time. ZERO contrast noise is injected
    # (the true α→∞ limit, not just a large finite α) — CEILING_ALPHA only labels
    # the SNR channel for a consistent report; it is not a claim that this
    # exposure is achievable. If the sweep is already close to this ceiling at a
    # realistic α, more exposure won't help — the model/architecture is the
    # bottleneck, not photon noise (and vice versa if there's a large gap).
    x_ceil = norm.encode(make_observable(raw_x, noise, obs_mode, alpha=CEILING_ALPHA))
    ceil_r2s = []
    for s in seeds:
        model, _ = _load_model(track, s, device)
        pred = lp.decode(_predict(model, x_ceil, device)).numpy()
        ceil_r2s.append(float(np.nanmean(metrics.report(y_true, pred, space="log10", bootstrap=0)["r2"])))
    ceiling = dict(r2_log10_mean=float(np.mean(ceil_r2s)), r2_log10_std=float(np.std(ceil_r2s)))

    provenance = _ckpt_provenance(track, seeds[0], device)
    _write_report(track, out_dir, seeds, per_seed_r2, rep, sweep, obs_mode, input_norm, cfg,
                  provenance, ceiling)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "track": track, "desc": cfg["desc"], "seeds": seeds, "n_seeds": len(seeds),
            "r2_log10_mean": float(np.mean(per_seed_r2)),
            "r2_log10_std": float(np.std(per_seed_r2)) if len(seeds) > 1 else None,
            "r2_log10_ensemble": float(np.nanmean(rep["r2"])),
            "rmse_dex": float(np.nanmean(rep["rmse"])),
            "mae_dex": float(np.nanmean(rep["mae"])),
            "per_species_r2": {c: float(rep["r2"][i]) for i, c in enumerate(TARGET_COLUMNS)},
            "run_config": provenance,
            "snr_sweep": sweep,
            "ceiling": ceiling,
        }, f, indent=2)
    _plot_sweep(track, out_dir, sweep, ceiling)
    _plot_scatter(track, out_dir, y_true, ens, rep)
    spread = f"± {np.std(per_seed_r2):.4f}" if len(seeds) > 1 else "(n=1)"
    print(f"=> {track}: test R²(log10) = {np.mean(per_seed_r2):.4f} {spread} "
          f"over seeds {seeds}; wrote {out_dir}")
    return rep, sweep


def _write_report(track, out_dir, seeds, per_seed_r2, rep, sweep, obs_mode, input_norm, cfg,
                  prov=None, ceiling=None):
    prov = prov or {}
    n = len(seeds)
    spread = (f"± {np.std(per_seed_r2):.4f}" if n > 1 else "(n=1; no std)")
    er, ep = prov.get("epochs_run"), prov.get("epochs_planned")
    undertrained = er is not None and ep is not None and er < ep
    L = []
    L.append("=" * 80)
    L.append("EXOPLANET ATMOSPHERIC RETRIEVAL — TEST-SET EVALUATION (ULTRAPLAN v2)")
    L.append("=" * 80)
    L.append(f"Track:            {track}  ({cfg['desc']})")
    L.append(f"Split:            FROZEN TEST (never used for model selection) — C3")
    L.append(f"Observable:       {obs_mode}   Input norm: {input_norm}   Labels: {cfg['label_transform']} (C1/H3/C2)")
    L.append(f"Seeds:            {seeds}  (n={n})")
    L.append(f"Ran:              {er}/{ep} epochs  | best@{prov.get('best_epoch')}  "
             f"| git {str(prov.get('git_commit'))[:12]}  | cfg {prov.get('config_hash')}")
    if undertrained or n < 3:
        flags = []
        if undertrained:
            flags.append(f"UNDERTRAINED ({er}/{ep} epochs)")
        if n < 3:
            flags.append(f"n={n} seed(s) (need ≥3 for CIs)")
        L.append(f"  ⚠ PRELIMINARY: {'; '.join(flags)} — not a citable result")
    L.append(f"Reporting space:  log10 mixing ratio (dex); R² NaN where target var=0 (honest, C2)")
    L.append("-" * 80)
    L.append(f"Overall test R² (log10), per-seed mean{'±std' if n > 1 else ''}:  "
             f"{np.mean(per_seed_r2):.4f} {spread}")
    L.append(f"Overall test R² (log10), seed-ensemble:      {np.nanmean(rep['r2']):.4f}")
    L.append(f"Overall RMSE (dex):  {np.nanmean(rep['rmse']):.4f}    Overall MAE (dex): {np.nanmean(rep['mae']):.4f}")
    L.append("-" * 80)
    L.append("PER-SPECIES (seed-ensemble, 95% bootstrap CI on R²):")
    L.append(f"{'Species':<9}| {'R2(log10)':<10}| {'95% CI':<20}| {'RMSE(dex)':<10}| {'MAE(dex)':<9}")
    L.append("-" * 80)
    for i, col in enumerate(TARGET_COLUMNS):
        ci = f"[{rep['r2_lo'][i]:.3f}, {rep['r2_hi'][i]:.3f}]"
        L.append(f"{col:<9}| {rep['r2'][i]:<10.4f}| {ci:<20}| {rep['rmse'][i]:<10.4f}| {rep['mae'][i]:<9.4f}")
    L.append("-" * 80)
    L.append("SNR SWEEP — C1 headline figure. α = √(t_exp/t_nominal); α=1 is the validated")
    L.append("LUVOIR-like nominal program (8 hr @ 5 pc, ∝d²; Zorzan+25 §3.1) — a photon-")
    L.append("starved floor (band planet SNR < 1 ⇒ R²≈0 expected BY PHYSICS, not a bug).")
    L.append(f"{'alpha':<8}| {'exposure ×':<11}| {'SNR_star(med)':<14}| {'SNR_planet(band)':<17}| {'R2(log10) mean±std':<22}")
    L.append("-" * 80)
    for r in sweep:
        L.append(f"{r['alpha']:<8}| {r['exposure_x']:<11.1f}| {r['snr_star_median']:<14.3f}| "
                 f"{r['snr_planet_band']:<17.3f}| {r['r2_log10_mean']:.4f} ± {r['r2_log10_std']:.4f}")
    if ceiling is not None:
        L.append("-" * 80)
        cr = ceiling["r2_log10_mean"]
        L.append(f"NOISELESS CEILING (zero injected contrast noise; the true α→∞ limit —")
        L.append(f"NOT a claim this exposure is achievable): R²(log10) = {cr:.4f} ± {ceiling['r2_log10_std']:.4f}")
        L.append("Fraction of this ceiling reached at each exposure in the sweep:")
        for r in sweep:
            if cr > 1e-6:
                frac = f"{100 * r['r2_log10_mean'] / cr:6.1f}%"
            else:
                frac = "  N/A (ceiling≈0)"
            L.append(f"  α={r['alpha']:<7}: {frac}")
        L.append("A small gap at high α means exposure time, not model/architecture capacity,")
        L.append("is the bottleneck; a large persistent gap means the reverse.")
    L.append("=" * 80)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "details_v2.txt"), "w") as f:
        f.write(txt)
    with open(os.path.join(out_dir, "snr_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader(); w.writerows(sweep)
    print(txt)


def _plot_sweep(track, out_dir, sweep, ceiling=None):
    """R² vs band-integrated planet SNR (physics axis, bottom) with the exposure-
    time multiple t/t_nom = α² on the top axis (mission-planning axis). Both are
    log-linear in α, so the twin axis is an exact relabeling, not a second plot."""
    xs = [r["snr_planet_band"] for r in sweep]
    ys = [r["r2_log10_mean"] for r in sweep]
    es = [r["r2_log10_std"] for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="navy")
    if ceiling is not None:
        ax.axhline(ceiling["r2_log10_mean"], color="darkorange", ls="--", alpha=0.8,
                   label=f"noiseless ceiling ({ceiling['r2_log10_mean']:.3f})")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xscale("log")
    ax.axvline(5.0, color="crimson", ls=":", alpha=0.7)
    ax.annotate("detection\nthreshold (SNR≈5)", xy=(5.0, ax.get_ylim()[0]),
                xytext=(4, 12), textcoords="offset points", fontsize=8, color="crimson")
    ax.set_xlabel("Band-integrated planet detection SNR (median, test set)")
    ax.set_ylabel("Overall R² (log10)")
    ax.grid(True, which="both", alpha=0.3)
    # top axis: exposure multiple. snr_band = c·α and exposure = α², so on log
    # scales the mapping is exact: exposure = (snr/c)².
    c = next(r["snr_planet_band"] / r["alpha"] for r in sweep)
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax2.set_xlim((lo / c) ** 2, (hi / c) ** 2)
    ax2.set_xlabel("Exposure multiple  $t/t_{\\rm nom}$  (nominal = 8 hr @ 5 pc, ∝$d^2$; Zorzan+25)")
    ax.set_title(f"{track}: retrieval accuracy vs planet SNR / exposure time (test set)", pad=32)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "r2_vs_snr.png"), dpi=150)
    plt.close(fig)


def _plot_scatter(track, out_dir, y_true, y_pred, rep):
    yt = np.log10(np.clip(y_true, 1e-12, None))
    yp = np.log10(np.clip(y_pred, 1e-12, None))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for i, ax in enumerate(axes.flatten()):
        ax.scatter(yt[:, i], yp[:, i], alpha=0.3, s=8, color="royalblue")
        lo = min(yt[:, i].min(), yp[:, i].min()); hi = max(yt[:, i].max(), yp[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_title(f"{TARGET_COLUMNS[i]}  R²={rep['r2'][i]:.3f}")
        ax.set_xlabel("true log10 X"); ax.set_ylabel("pred log10 X")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_true_log10.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Test-set evaluator + SNR sweep (ULTRAPLAN).")
    ap.add_argument("track")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    a = ap.parse_args()
    evaluate_track(a.track, seeds=a.seeds, cache_dir=a.cache, alphas=a.alphas)


if __name__ == "__main__":
    main()
