"""Quantify cloud robustness: original (clear-trained) vs cloud-robust model,
each on clear and cloudy validation spectra.

For every track it prints/saves a 2x2 table of average R^2:

                     clear val      cloudy val (f=0.7)
    original model   <baseline>     <expected drop>       <- Option-1 diagnosis
    cloud-robust     <retain>       <expected recovery>

plus a per-species breakdown and an R^2-vs-cloud-opacity degradation curve.

Each model is fed inputs standardized with ITS OWN training stats (original
caches for the original models; cloudaug stats -- or cache_planet for causal --
for the robust models), exactly like the existing crosstest.py. Predictions are
inverse-transformed to linear abundance with the model's label stats; truth is
the raw val_y. (R^2 is invariant to the per-column affine standardization, so
clear/cloudy/model comparisons are apples-to-apples.)

ISOLATION: every checkpoint and cache is opened read-only.

    python eval_cloud_robustness.py                 # all tracks
    python eval_cloud_robustness.py original1dcnn    # one track
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from cloud_model import apply_cloud  # noqa: E402
from cloud_dataset import make_cloudy_val  # noqa: E402
from train_common import _load_model_class  # noqa: E402

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
SRC = os.path.join(REPO, "src")
DATA = os.path.join(REPO, "data")

TARGETS = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2']

# Opacity sweep for the degradation curve; (f, b). f=0 is clear.
SWEEP = [(0.0, 0.0), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5), (1.0, 0.5)]
HEADLINE_CLOUD = (0.7, 0.5)   # the "cloudy val" column in the 2x2
EVAL_N = 6000                 # subsample of the val split for speed
SEED = 123

EVAL = {
    "original1dcnn": dict(
        model_py=f"{SRC}/original1dcnn/model.py", cls="NasaInaraModel",
        base_cache="cache_original", cloudaug="cache_cloudaug_original",
        orig_ckpt=f"{SRC}/original1dcnn/checkpoints/model_best.pth", orig_stats="cache_original",
        robust_ckpt=f"{THIS_DIR}/original1dcnn/checkpoints/model_best.pth", robust_stats="cache_cloudaug_original",
    ),
    "optimized1dcnn": dict(
        model_py=f"{SRC}/optimized1dcnn/model.py", cls="NasaInaraModel",
        base_cache="cache", cloudaug="cache_cloudaug_both",
        orig_ckpt=f"{SRC}/optimized1dcnn/checkpoints/model_best.pth", orig_stats="cache",
        robust_ckpt=f"{THIS_DIR}/optimized1dcnn/checkpoints/model_best.pth", robust_stats="cache_cloudaug_both",
    ),
    "2channel1dcnn": dict(
        model_py=f"{SRC}/2channel1dcnn/model.py", cls="NasaInaraModel",
        base_cache="cache", cloudaug="cache_cloudaug_both",
        orig_ckpt=f"{SRC}/2channel1dcnn/checkpoints/model_best.pth", orig_stats="cache",
        robust_ckpt=f"{THIS_DIR}/2channel1dcnn/checkpoints/model_best.pth", robust_stats="cache_cloudaug_both",
    ),
    "transformerarch": dict(
        model_py=f"{SRC}/transformerarch/model.py", cls="NasaInaraTransformer",
        base_cache="cache_planet", cloudaug="cache_cloudaug_planet",
        orig_ckpt=f"{SRC}/transformerarch/checkpoints/model_best.pth", orig_stats="cache_planet",
        robust_ckpt=f"{THIS_DIR}/transformerarch/checkpoints/model_best.pth", robust_stats="cache_cloudaug_planet",
    ),
    "causal": dict(
        model_py=f"{SRC}/causal/cnn_trnas/model.py", cls="NasaInaraTransformer",
        base_cache="cache_planet", cloudaug="cache_cloudaug_planet",
        orig_ckpt=f"{SRC}/causal/cnn_trnas/checkpoints/model_best.pth", orig_stats="cache_planet",
        robust_ckpt=f"{THIS_DIR}/causal/checkpoints/model_best.pth", robust_stats="cache_planet",
    ),
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


def _r2_per_species(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # C2: R2 undefined when ss_tot==0; no epsilon bias
        return np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)


def _load_stats(cache_name):
    d = os.path.join(DATA, cache_name)
    return (torch.load(os.path.join(d, "mean_x.pt"), map_location="cpu"),
            torch.load(os.path.join(d, "std_x.pt"), map_location="cpu"),
            torch.load(os.path.join(d, "mean_y.pt"), map_location="cpu"),
            torch.load(os.path.join(d, "std_y.pt"), map_location="cpu"))


def _predict(model, raw_x, mean_x, std_x, mean_y, std_y, device, batch=256):
    """Standardize raw_x with the model's input stats, predict, inverse-transform
    to linear abundance with the model's label stats."""
    std_in = ((raw_x - mean_x) / (std_x + 1e-30)).float()
    preds = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, std_in.shape[0], batch):
            xb = std_in[lo:lo + batch].to(device)
            preds.append(model(xb).cpu())
    pred_std = torch.cat(preds, dim=0)
    pred_lin = pred_std * std_y + mean_y
    return pred_lin.numpy()


def _load_model(model_py, cls, ckpt_path, in_ch, seq_len, device):
    """Returns (model, best_epoch). best_epoch is the epoch the loaded
    checkpoint (model_best.pth) was saved at, or None if unavailable."""
    if not os.path.exists(ckpt_path):
        return None, None
    ModelClass = _load_model_class(model_py, cls)
    model = ModelClass(in_channels=in_ch, sequence_length=seq_len).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["state_dict"])
    return model, ck.get("epoch")


def evaluate_track(track):
    cfg = EVAL[track]
    device = _device()
    out_dir = os.path.join(THIS_DIR, track)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Cloud-robustness eval: {track} on {device} ===")

    base = os.path.join(DATA, cfg["base_cache"])
    val_x = torch.load(os.path.join(base, "val_x.pt"), map_location="cpu")
    val_y_raw = torch.load(os.path.join(base, "val_y.pt"), map_location="cpu")
    val_cont = torch.load(os.path.join(DATA, cfg["cloudaug"], "val_continuum.pt"), map_location="cpu")

    # Subsample for speed (fixed seed).
    n = min(EVAL_N, val_x.shape[0])
    idx = torch.randperm(val_x.shape[0], generator=torch.Generator().manual_seed(SEED))[:n]
    val_x, val_y_raw, val_cont = val_x[idx], val_y_raw[idx], val_cont[idx]
    in_ch, seq_len = val_x.shape[1], val_x.shape[2]
    truth = val_y_raw.numpy()
    print(f"  val subset {n}  channels={in_ch} seq_len={seq_len}")

    # Pre-generate the condition spectra (raw) once; reused by both models.
    conditions = {}
    for f, b in SWEEP:
        conditions[f] = val_x if f == 0.0 else make_cloudy_val(val_x, val_cont, f, b)

    # Load both models (and the epoch each best-checkpoint was saved at).
    models = {}
    s_orig = _load_stats(cfg["orig_stats"])
    s_rob = _load_stats(cfg["robust_stats"])
    m_orig, ep_orig = _load_model(cfg["model_py"], cfg["cls"], cfg["orig_ckpt"], in_ch, seq_len, device)
    m_rob, ep_rob = _load_model(cfg["model_py"], cfg["cls"], cfg["robust_ckpt"], in_ch, seq_len, device)
    models["original"] = (m_orig, s_orig)
    models["robust"] = (m_rob, s_rob)
    epochs = {"original": ep_orig, "robust": ep_rob}

    if m_orig is None:
        print(f"  [skip] original checkpoint not found: {cfg['orig_ckpt']}")
    if m_rob is None:
        print(f"  [warn] robust checkpoint not found: {cfg['robust_ckpt']} (train it first)")

    # results[model][f] = (avg_r2, per_species_r2)
    results = {m: {} for m in models}
    for mname, (model, (mx, sx, my, sy)) in models.items():
        if model is None:
            continue
        for f, _b in SWEEP:
            pred = _predict(model, conditions[f], mx, sx, my, sy, device)
            r2 = _r2_per_species(truth, pred)
            results[mname][f] = (float(np.mean(r2)), r2)

    _write_report(track, out_dir, results, epochs)
    _plot(track, out_dir, results)
    print(f"=> Wrote details.txt to {out_dir} and charts to {os.path.join(out_dir, 'charts')}")
    return results


def _fmt(results, m, f):
    if m in results and f in results[m]:
        return f"{results[m][f][0]:+.4f}"
    return "   n/a "


def _write_report(track, out_dir, results, epochs=None):
    hf = HEADLINE_CLOUD[0]
    lines = []
    lines.append("=" * 78)
    lines.append(f"CLOUD ROBUSTNESS REPORT — {track}")
    lines.append("=" * 78)
    if epochs:
        eo = epochs.get("original")
        er = epochs.get("robust")
        eo_s = str(eo) if eo is not None else "n/a"
        er_s = str(er) if er is not None else "n/a"
        lines.append(f"Num epochs (best checkpoint):  original={eo_s}   cloud-robust={er_s}")
    lines.append(f"Eval val subset: {EVAL_N} planets   headline cloudy f={hf}")
    lines.append("")
    lines.append(f"Average R^2 (2x2):    clear (f=0)     cloudy (f={hf})")
    lines.append("-" * 78)
    lines.append(f"  original model   |   {_fmt(results,'original',0.0)}      |   {_fmt(results,'original',hf)}")
    lines.append(f"  cloud-robust     |   {_fmt(results,'robust',0.0)}      |   {_fmt(results,'robust',hf)}")
    lines.append("-" * 78)
    # Deltas
    try:
        drop = results['original'][0.0][0] - results['original'][hf][0]
        lines.append(f"  original clear->cloudy drop:   {drop:+.4f}  (the cloud-blindness)")
    except KeyError:
        pass
    try:
        rec = results['robust'][hf][0] - results['original'][hf][0]
        lines.append(f"  robust vs original on cloudy:  {rec:+.4f}  (the robustness gain)")
    except KeyError:
        pass
    lines.append("")
    lines.append("Average R^2 vs cloud opacity f:")
    lines.append("  f      " + "  ".join(f"{f:>6.1f}" for f, _ in SWEEP))
    for m in ("original", "robust"):
        if results.get(m):
            row = "  ".join(f"{results[m][f][0]:>6.3f}" if f in results[m] else "   n/a" for f, _ in SWEEP)
            lines.append(f"  {m:<7}" + row)
    lines.append("")
    lines.append(f"Per-species R^2 at cloudy (f={hf}):")
    lines.append(f"  {'species':<8} | {'original':>9} | {'robust':>9} | {'delta':>8}")
    lines.append("-" * 50)
    for i, sp in enumerate(TARGETS):
        o = results['original'][hf][1][i] if results.get('original', {}).get(hf) else float('nan')
        r = results['robust'][hf][1][i] if results.get('robust', {}).get(hf) else float('nan')
        lines.append(f"  {sp:<8} | {o:>9.4f} | {r:>9.4f} | {r-o:>8.4f}")
    lines.append("=" * 78)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "cloud_robustness_details.txt"), "w") as fh:
        fh.write(report + "\n")


def _plot(track, out_dir, results):
    hf = HEADLINE_CLOUD[0]
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    # (1) degradation curve
    plt.figure(figsize=(9, 6))
    fs = [f for f, _ in SWEEP]
    for m, color in (("original", "tab:red"), ("robust", "tab:blue")):
        if results.get(m):
            ys = [results[m][f][0] if f in results[m] else np.nan for f in fs]
            plt.plot(fs, ys, marker="o", color=color, label=m)
    plt.xlabel("cloud opacity  f"); plt.ylabel("average R^2")
    plt.title(f"{track}: retrieval accuracy vs cloud opacity")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(charts_dir, "cloud_degradation_curve.png"), dpi=150)
    plt.close()

    # (2) per-species grouped bar at cloudy
    if results.get('original', {}).get(hf) and results.get('robust', {}).get(hf):
        o = results['original'][hf][1]; r = results['robust'][hf][1]
        x = np.arange(len(TARGETS)); w = 0.4
        plt.figure(figsize=(14, 6))
        plt.bar(x - w/2, o, w, label="original", color="tab:red", alpha=0.8)
        plt.bar(x + w/2, r, w, label="cloud-robust", color="tab:blue", alpha=0.8)
        plt.axhline(0, color="k", lw=0.6)
        plt.xticks(x, TARGETS, rotation=45); plt.ylabel("R^2")
        plt.title(f"{track}: per-species R^2 on cloudy spectra (f={hf})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "cloud_per_species_r2.png"), dpi=150)
        plt.close()


def main():
    tracks = sys.argv[1:] or list(EVAL.keys())
    for t in tracks:
        if t not in EVAL:
            raise SystemExit(f"Unknown track '{t}'. Choose from {list(EVAL)}.")
        evaluate_track(t)


if __name__ == "__main__":
    main()
