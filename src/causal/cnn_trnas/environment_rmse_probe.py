#!/usr/bin/env python
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Set path relative to this script
DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))

from dataloader import InaraDataset  # noqa: E402
from model import NasaInaraTransformer  # noqa: E402

SUMMARY_PATH = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/summary.csv"
SPECTRA_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/inara_1by3"
CACHE_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/cache_planet"
CHECKPOINT_PATH = DIR / "checkpoints" / "model_best.pth"
OUT_DIR = DIR / "environment_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2']
ENV_COLUMNS = ["star_class", "star_temperature", "distance_parsec", "surface_temperature", "surface_pressure"]

class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return idx, x, y

def env_label(summary, planet_idx, col):
    row = summary.loc[planet_idx]
    v = row[col]
    if col == "star_class":
        return str(v)
    return str(row[f"{col}_bin"])

def rmse(y_true, y_pred):
    per_target = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return float(np.mean(per_target))

def main():
    max_samples = 1024
    summary = pd.read_csv(SUMMARY_PATH, dtype={"planet_index": str})
    summary["planet_index_int"] = summary["planet_index"].astype(int)
    summary = summary.set_index("planet_index_int")

    for col in [c for c in ENV_COLUMNS if c != "star_class"]:
        summary[f"{col}_bin"] = pd.qcut(summary[col], q=4, duplicates="drop").astype(str)

    base = InaraDataset(str(SUMMARY_PATH), str(SPECTRA_DIR), feature_mode="planet")
    n_total = len(base)
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_size = int(0.8 * n_total)
    val_indices = indices[train_size:]
    val_indices = val_indices[: min(max_samples, len(val_indices))]

    mean_x = torch.load(os.path.join(CACHE_DIR, "mean_x.pt"), map_location="cpu")
    std_x = torch.load(os.path.join(CACHE_DIR, "std_x.pt"), map_location="cpu")
    mean_y = torch.load(os.path.join(CACHE_DIR, "mean_y.pt"), map_location="cpu").numpy()
    std_y = torch.load(os.path.join(CACHE_DIR, "std_y.pt"), map_location="cpu").numpy()

    wrapped = IndexedDataset(base)
    loader = DataLoader(Subset(wrapped, val_indices), batch_size=64, shuffle=False)

    _, x0, _ = wrapped[val_indices[0]]
    model = NasaInaraTransformer(in_channels=x0.shape[0], sequence_length=x0.shape[1])
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    records = []
    with torch.no_grad():
        for idxs, x, y_raw in loader:
            x = (x - mean_x) / (std_x + 1e-30)
            pred_std = model(x).numpy()
            pred_raw = pred_std * std_y + mean_y
            y_raw = y_raw.numpy()
            for local_i, dataset_idx in enumerate(idxs.tolist()):
                fname = base.file_names[dataset_idx]
                planet_idx = int(fname.split(".")[0])
                records.append({
                    "planet_idx": planet_idx,
                    "y_true": y_raw[local_i],
                    "y_pred": pred_raw[local_i],
                    "envs": {col: env_label(summary, planet_idx, col) for col in ENV_COLUMNS},
                })

    overall = rmse(np.stack([r["y_true"] for r in records]), np.stack([r["y_pred"] for r in records]))
    rows = []
    for col in ENV_COLUMNS:
        buckets = defaultdict(list)
        for r in records:
            buckets[r["envs"][col]].append(r)
        for label, items in buckets.items():
            if len(items) < 10:
                continue
            val = rmse(np.stack([r["y_true"] for r in items]), np.stack([r["y_pred"] for r in items]))
            rows.append({"environment_variable": col, "environment": label, "n": len(items), "rmse": val, "gap_vs_overall": val - overall})
    rows = sorted(rows, key=lambda r: r["rmse"], reverse=True)

    result = {
        "n_eval_samples": len(records),
        "overall_rmse": overall,
        "worst_environment": rows[0] if rows else None,
        "rows": rows,
    }
    with open(OUT_DIR / "environment_rmse_results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(OUT_DIR / "environment_rmse_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["environment_variable", "environment", "n", "rmse", "gap_vs_overall"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
