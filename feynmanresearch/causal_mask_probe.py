#!/usr/bin/env python
"""No-source-edit causal/masking probe for the transformer-CNN model.

This is not proof of causality. It measures how prediction error changes when
future/right-side wavelength positions are masked, approximating whether the
model depends on full-sequence context versus prefix-local spectral evidence.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path("/Users/aakashrajput/MachineLearning/Exoplanets")
SRC = REPO / "src" / "transformerarch"
sys.path.insert(0, str(SRC))
os.chdir(REPO)

# CPU only: avoids the MPS/torchinfo side-effect seen in src/transformerarch/test.py.
torch.backends.mps.is_available = lambda: False

from dataloader import load_cached_data  # noqa: E402
from model import NasaInaraTransformer  # noqa: E402

SUMMARY_PATH = REPO / "data" / "summary.csv"
SPECTRA_DIR = REPO / "data" / "inara_1by3"
CACHE_DIR = REPO / "data" / "cache_planet"
CHECKPOINT_PATH = SRC / "checkpoints" / "model_best.pth"
OUT_DIR = REPO / "feynmanresearch" / "mask_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    'H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O',
    'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2'
]


def inverse_y(y):
    mean_y = torch.load(CACHE_DIR / "mean_y.pt", map_location="cpu").numpy()
    std_y = torch.load(CACHE_DIR / "std_y.pt", map_location="cpu").numpy()
    return y * std_y + mean_y


def eval_with_keep_fraction(model, loader, keep_fraction):
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.clone()
            seq = x.shape[-1]
            keep = int(round(seq * keep_fraction))
            if keep < seq:
                # Mask the right/future side in standardized space: zero means train mean.
                x[:, :, keep:] = 0.0
            pred = model(x)
            preds.append(pred.numpy())
            trues.append(y.numpy())
    y_pred = inverse_y(np.concatenate(preds, axis=0))
    y_true = inverse_y(np.concatenate(trues, axis=0))
    per_target_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return {
        "overall_rmse": float(np.mean(per_target_rmse)),
        "per_target_rmse": {k: float(v) for k, v in zip(TARGET_COLUMNS, per_target_rmse)},
    }


def main():
    train_dataset, val_dataset = load_cached_data(
        str(CACHE_DIR), str(SUMMARY_PATH), str(SPECTRA_DIR),
        normalize_inputs=True, feature_mode="planet"
    )
    n = min(512, len(val_dataset))
    subset = Subset(val_dataset, range(n))
    loader = DataLoader(subset, batch_size=64, shuffle=False)

    in_channels = val_dataset[0][0].shape[0]
    seq_len = val_dataset[0][0].shape[1]
    model = NasaInaraTransformer(in_channels=in_channels, sequence_length=seq_len)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    keep_fractions = [1.0, 0.75, 0.50, 0.25]
    results = {
        "n_eval_samples": n,
        "sequence_length": seq_len,
        "interpretation": "Right-side wavelength masking probe; lower RMSE is better. This is an occlusion/robustness proxy, not proof of causality.",
        "results": {}
    }
    for frac in keep_fractions:
        results["results"][str(frac)] = eval_with_keep_fraction(model, loader, frac)

    # Derived sensitivity: larger increase means stronger dependence on masked suffix.
    base = results["results"]["1.0"]["overall_rmse"]
    for frac in keep_fractions[1:]:
        val = results["results"][str(frac)]["overall_rmse"]
        results["results"][str(frac)]["rmse_increase_vs_full"] = float(val - base)

    with open(OUT_DIR / "mask_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT_DIR / "mask_probe_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keep_fraction", "overall_rmse", "rmse_increase_vs_full"])
        for frac in keep_fractions:
            r = results["results"][str(frac)]
            w.writerow([frac, r["overall_rmse"], r.get("rmse_increase_vs_full", 0.0)])

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
