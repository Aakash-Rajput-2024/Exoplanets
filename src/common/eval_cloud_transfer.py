"""Held-out cloud-family transfer test (fixes C5).

Loads a track's matched-budget v2 checkpoints and scores them on the FROZEN TEST
split under clear conditions and under each cloud family. `grey` is the family a
cloud-augmented model would have trained on (the "seen" reference); `non_grey`,
`band_selective`, `patchy` are structurally different and **held out**. The gap

    Δ = R²(clear) − R²(family)

is the non-circular robustness measurement C5 demands. Run it on a clean-trained
model to quantify raw cloud sensitivity, or on a grey-cloud-augmented model
(train_runner --cloud grey) to read the seen-vs-unseen generalization gap.

    python -m common.eval_cloud_transfer <track> [--seeds ...] [--suffix _grey]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.registry import track_config, CACHE_V2
from common.runtime import get_device
from common.data import load_wavelength
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED
from common.inputs import build_eval_observable
from common import metrics, cloud_families
from common.evaluate import _load_model, _predict, TARGET_COLUMNS

CONDITIONS = ("clear", "grey", "non_grey", "band_selective", "patchy")


def _ckpt(track, seed, suffix):
    d = os.path.dirname(track_config(track)["model_py"])
    p_track = os.path.join(d, "checkpoints_v2", f"model_best_{track}_seed{seed}{suffix}.pth")
    if os.path.exists(p_track):
        return p_track
    return os.path.join(d, "checkpoints_v2", f"model_best_seed{seed}{suffix}.pth")


def evaluate_cloud_transfer(track, seeds=None, cache_dir=CACHE_V2, suffix=""):
    cfg = track_config(track)
    seeds = list(seeds) if seeds is not None else list(cfg["seeds"])
    seeds = [s for s in seeds if os.path.exists(_ckpt(track, s, suffix))]
    if not seeds:
        raise SystemExit(f"No checkpoints model_best_seed*{suffix}.pth for {track}")
    device = get_device()
    out_dir = os.path.join(os.path.dirname(cfg["model_py"]), "eval_v2")
    os.makedirs(out_dir, exist_ok=True)

    raw_x, y_lin, noise, _ids, lp = load_eval_raw(cache_dir, "test", cfg["label_transform"])
    y_true = y_lin.numpy()
    wl = load_wavelength(cache_dir).numpy()             # tail-trimmed to match raw_x
    obs_mode, input_norm = cfg["obs_mode"], cfg["input_norm"]
    norm = get_norm(cache_dir, obs_mode, input_norm)

    star = (raw_x[:, 0, :] - raw_x[:, 1, :])            # (N,L) F_star (clouds don't touch it)
    planet = raw_x[:, 1, :].numpy().astype(np.float64)   # (N,L)
    print(f"  estimating test planet continuum ({planet.shape}) ...")
    cont = cloud_families.estimate_continuum(planet)

    # Pre-load models once.
    models = [_load_model(track, s, device, suffix=suffix)[0] for s in seeds]

    rows = []
    for cond in CONDITIONS:
        p_cond = planet if cond == "clear" else cloud_families.apply_family(cond, planet, cont, wl)
        p_cond_t = torch.from_numpy(np.asarray(p_cond)).float()
        # Rebuild the star+planet channel from the CLOUDY planet so the star-only
        # SNR stays clean and the contrast is consistent (clouds modify F_p only).
        sp_cond = (star + p_cond_t).unsqueeze(1)
        x_cond = torch.cat([sp_cond, p_cond_t.unsqueeze(1)], dim=1)
        x_obs = build_eval_observable(x_cond, noise, obs_mode, norm, alpha=1.0, seed=TEST_NOISE_SEED)
        r2s = []
        for model in models:
            pred = lp.decode(_predict(model, x_obs, device)).numpy()
            r2s.append(float(np.nanmean(metrics.report(y_true, pred, space="log10", bootstrap=0)["r2"])))
        rows.append(dict(family=cond, r2_log10_mean=float(np.mean(r2s)),
                         r2_log10_std=float(np.std(r2s))))
        print(f"  {cond:<15} R²(log10) = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    _write(track, out_dir, seeds, rows, suffix, cfg)
    _plot(track, out_dir, rows, suffix)
    return rows


def _write(track, out_dir, seeds, rows, suffix, cfg):
    clear = next(r for r in rows if r["family"] == "clear")["r2_log10_mean"]
    L = ["=" * 78,
         f"HELD-OUT CLOUD-FAMILY TRANSFER — {track}{suffix or ' (clean-trained)'}",
         "=" * 78,
         f"Split: FROZEN TEST | seeds {seeds} | obs={cfg['obs_mode']} labels={cfg['label_transform']}",
         "grey = training family (seen ref); non_grey/band_selective/patchy = HELD OUT (C5)",
         "-" * 78,
         f"{'family':<16}| {'R2(log10)':<20}| {'gap vs clear':<12}"]
    L.append("-" * 78)
    for r in rows:
        gap = clear - r["r2_log10_mean"]
        held = "  (HELD OUT)" if r["family"] in cloud_families.HELD_OUT_FAMILIES else ""
        L.append(f"{r['family']:<16}| {r['r2_log10_mean']:.4f} ± {r['r2_log10_std']:.4f}      "
                 f"| {gap:+.4f}{held}")
    L.append("=" * 78)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, f"cloud_transfer{suffix}.txt"), "w") as f:
        f.write(txt)
    with open(os.path.join(out_dir, f"cloud_transfer{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(txt)


def _plot(track, out_dir, rows, suffix):
    names = [r["family"] for r in rows]
    ys = [r["r2_log10_mean"] for r in rows]
    es = [r["r2_log10_std"] for r in rows]
    colors = ["gray" if n == "clear" else "tab:green" if n == "grey" else "tab:red" for n in names]
    plt.figure(figsize=(9, 5))
    plt.bar(names, ys, yerr=es, capsize=4, color=colors, alpha=0.85)
    plt.axhline(0, color="k", lw=0.6)
    plt.ylabel("Overall R² (log10)")
    plt.title(f"{track}{suffix}: cloud-family transfer (green=seen grey, red=held out)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"cloud_transfer{suffix}.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Held-out cloud-family transfer (ULTRAPLAN C5).")
    ap.add_argument("track")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--suffix", default="", help="checkpoint suffix, e.g. _grey")
    a = ap.parse_args()
    evaluate_cloud_transfer(a.track, seeds=a.seeds, cache_dir=a.cache, suffix=a.suffix)


if __name__ == "__main__":
    main()
