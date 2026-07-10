"""Shared cross-generator evaluation runner (torch side).

Each model directory's crosstest.py is a thin wrapper that constructs its model
and calls run_crosstest(). This mirrors the existing test.py flow but:
  - loads synthetic spectra from a .npy cross-gen cache instead of the INARA cache,
  - standardizes inputs and un-standardizes targets with INARA *training* stats
    (so we test the trained weights fairly, not stats refit to synthetic data),
  - reports the headline metric over the species the engine actually modeled
    ("covered"), since un-covered species are zeroed and uninformative.

Run in the PyTorch env used for test.py (e.g. venvNLP).
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

INARA_TARGETS = [
    "H2O", "CO2", "O2", "N2", "CH4", "N2O",
    "CO", "O3", "SO2", "NH3", "C2H6", "NO2",
]


def _metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # C2: R2 undefined when ss_tot==0; no epsilon bias
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return r2, rmse, mae


def _load_inara_stats(stats_dir):
    return {k: torch.load(os.path.join(stats_dir, f"{k}.pt")).cpu().numpy()
            for k in ("mean_x", "std_x", "mean_y", "std_y")}


def run_crosstest(model, checkpoint_path, cache_dir, inara_stats_dir,
                  out_dir, model_name, device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    X = np.load(os.path.join(cache_dir, "val_x.npy"))   # [N, C, L] raw flux
    Y = np.load(os.path.join(cache_dir, "val_y.npy"))   # [N, 12] linear abundances
    manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
    print(f"cross-gen cache: {manifest['engine']} {manifest['feature_mode']} "
          f"N={X.shape[0]} C={X.shape[1]} L={X.shape[2]}")

    stats = _load_inara_stats(inara_stats_dir)
    X_std = (X - stats["mean_x"]) / (stats["std_x"] + 1e-30)  # INARA channel stats

    model = model.to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print("Loaded model from checkpoint.")
    else:
        print("WARNING: no checkpoint found; evaluating random weights.")

    model.eval()
    preds = []
    xb = torch.tensor(X_std, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(xb), 64):
            preds.append(model(xb[i:i + 64].to(device)).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) * stats["std_y"] + stats["mean_y"]  # -> linear
    y_true = Y

    r2, rmse, mae = _metrics(y_true, y_pred)

    covered = [s for s in manifest["covered_species"] if s in INARA_TARGETS]
    cov_idx = [INARA_TARGETS.index(s) for s in covered]

    lines = [
        "=" * 80,
        "CROSS-GENERATOR EVALUATION REPORT",
        "=" * 80,
        f"Model: {model_name}",
        f"Eval generator: {manifest['engine']}  (training generator: NASA PSG / INARA)",
        f"Feature mode: {manifest['feature_mode']}  | Channels: {X.shape[1]}  | Seq len: {X.shape[2]}",
        f"Samples: {X.shape[0]}",
        f"Covered species (modeled by engine): {', '.join(covered)}",
        "-" * 80,
        "HEADLINE (covered species only):",
        f"  Avg R2:   {np.mean(r2[cov_idx]):.4f}",
        f"  Avg RMSE: {np.mean(rmse[cov_idx]):.4f}",
        f"  Avg MAE:  {np.mean(mae[cov_idx]):.4f}",
        "-" * 80,
        "ALL 12 SPECIES (uncovered species are zeroed -> uninformative):",
        f"{'Species':<8} | {'Covered':<7} | {'R2':<9} | {'RMSE':<9} | {'MAE':<9}",
        "-" * 80,
    ]
    for i, sp in enumerate(INARA_TARGETS):
        flag = "yes" if sp in covered else "-"
        lines.append(f"{sp:<8} | {flag:<7} | {r2[i]:<9.4f} | {rmse[i]:<9.4f} | {mae[i]:<9.4f}")
    lines.append("=" * 80)
    report = "\n".join(lines) + "\n"

    suffix = manifest["engine"]
    details_path = os.path.join(out_dir, f"crossgen_details_{suffix}.txt")
    with open(details_path, "w") as f:
        f.write(report)
    print(report)
    print(f"=> wrote {details_path}")

    charts_dir = os.path.join(out_dir, f"crossgen_charts_{suffix}")
    os.makedirs(charts_dir, exist_ok=True)
    _plot_pred_vs_true(y_true, y_pred, r2, rmse, covered, charts_dir)
    _plot_residual_box(y_pred - y_true, charts_dir)
    print(f"=> wrote charts to {charts_dir}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "covered": covered}


def _plot_pred_vs_true(y_true, y_pred, r2, rmse, covered, charts_dir):
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, sp in enumerate(INARA_TARGETS):
        ax = axes[i]
        is_cov = sp in covered
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.4,
                   color="royalblue" if is_cov else "lightgray")
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        tag = "" if is_cov else " (uncovered)"
        ax.set_title(f"{sp}{tag}\nR2={r2[i]:.3f} | RMSE={rmse[i]:.3f}")
        ax.set_xlabel("True abundance")
        ax.set_ylabel("Predicted abundance")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "predicted_vs_true.png"), dpi=150)
    plt.close()


def _plot_residual_box(residuals, charts_dir):
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=residuals, palette="Set3")
    plt.xticks(ticks=range(12), labels=INARA_TARGETS, rotation=45)
    plt.axhline(0, color="red", linestyle="--", alpha=0.7)
    plt.title("Cross-generator residuals across species (Predicted - True)")
    plt.ylabel("Residual (abundance)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "residuals_boxplot.png"), dpi=150)
    plt.close()
