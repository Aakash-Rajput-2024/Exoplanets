"""Shared training core for the cloud-robust tracks.

Each track in `src/cloud_robust/<track>/train.py` is a thin wrapper that calls
`train_track("<track>")`. This module:
  - imports each track's ORIGINAL model class by file path (no duplication, no
    module-name collisions),
  - builds the on-the-fly cloud-augmented training set (+ counterfactual set for
    the causal track),
  - runs a generic loop that mirrors each track's hyperparameters
    (loss / optimizer / scheduler / grad-clip / early stopping),
  - writes checkpoints under src/cloud_robust/<track>/checkpoints/.

ISOLATION: base caches, original checkpoints, and original training code are
read-only. Nothing here writes outside src/cloud_robust/ and data/cache_cloudaug_*.
"""

from __future__ import annotations

import os
import sys
import math
import importlib.util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)
from cloud_dataset import CloudAugmentedDataset

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
SRC = os.path.join(REPO, "src")
DATA = os.path.join(REPO, "data")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, desc=""):
        return it


# ---------------------------------------------------------------------------
# Track registry: mirrors each original track's setup exactly.
# stats_cache=None means "use the cloudaug cache's augmented stats".
# ---------------------------------------------------------------------------
TRACKS = {
    "original1dcnn": dict(
        model_py=f"{SRC}/original1dcnn/model.py", model_cls="NasaInaraModel",
        base_cache="cache_original", cloudaug="cache_cloudaug_original", stats_cache=None,
        batch=1024, lr=1e-3, epochs=50, patience=10,
        loss="l1", scheduler=None, grad_clip=None, counterfactual=False,
    ),
    "optimized1dcnn": dict(
        model_py=f"{SRC}/optimized1dcnn/model.py", model_cls="NasaInaraModel",
        base_cache="cache", cloudaug="cache_cloudaug_both", stats_cache=None,
        batch=1024, lr=1e-3, epochs=50, patience=10,
        loss="l1", scheduler=None, grad_clip=None, counterfactual=False,
    ),
    "2channel1dcnn": dict(
        model_py=f"{SRC}/2channel1dcnn/model.py", model_cls="NasaInaraModel",
        base_cache="cache", cloudaug="cache_cloudaug_both", stats_cache=None,
        batch=1024, lr=1e-3, epochs=50, patience=10,
        loss="l1", scheduler=None, grad_clip=None, counterfactual=False,
    ),
    "transformerarch": dict(
        model_py=f"{SRC}/transformerarch/model.py", model_cls="NasaInaraTransformer",
        base_cache="cache_planet", cloudaug="cache_cloudaug_planet", stats_cache=None,
        batch=16, lr=1e-3, epochs=50, patience=15, warmup_epochs=5,
        loss="mse", scheduler="cosine_warmup", grad_clip=1.0, counterfactual=False,
    ),
    # Causal: combine cloud augmentation with the existing env-counterfactual set.
    # Both halves standardized with ORIGINAL cache_planet stats (the counterfactual
    # tensors are already in that space), so stats_cache="cache_planet".
    "causal": dict(
        model_py=f"{SRC}/causal/cnn_trnas/model.py", model_cls="NasaInaraTransformer",
        base_cache="cache_planet", cloudaug="cache_cloudaug_planet", stats_cache="cache_planet",
        batch=16, lr=1e-3, epochs=50, patience=15, warmup_epochs=5,
        loss="mse", scheduler="cosine_warmup", grad_clip=1.0, counterfactual=True,
        cf_x=f"{SRC}/causal/cnn_trnas/checkpoints/train_x_augmented.pt",
        cf_y=f"{SRC}/causal/cnn_trnas/checkpoints/train_y_augmented.pt",
    ),
}

N_CLOUDY = 2
SEED = 7


def _load_model_class(model_py: str, cls_name: str):
    """Load a uniquely-named module from a file path and return the class."""
    mod_name = "cr_" + os.path.basename(os.path.dirname(model_py)) + "_model"
    spec = importlib.util.spec_from_file_location(mod_name, model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def _cosine_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup = warmup_epochs * steps_per_epoch
    total = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_cloud_dataset(cfg):
    base = os.path.join(DATA, cfg["base_cache"])
    aug = os.path.join(DATA, cfg["cloudaug"])
    stats = os.path.join(DATA, cfg["stats_cache"]) if cfg["stats_cache"] else aug

    raw_x = torch.load(os.path.join(base, "train_x.pt"), map_location="cpu")
    raw_y = torch.load(os.path.join(base, "train_y.pt"), map_location="cpu")
    cont = torch.load(os.path.join(aug, "continuum_train.pt"), map_location="cpu")
    mean_x = torch.load(os.path.join(stats, "mean_x.pt"), map_location="cpu")
    std_x = torch.load(os.path.join(stats, "std_x.pt"), map_location="cpu")
    mean_y = torch.load(os.path.join(stats, "mean_y.pt"), map_location="cpu")
    std_y = torch.load(os.path.join(stats, "std_y.pt"), map_location="cpu")

    # 2ch original-cache stats are stored as (1,C,1); 1ch likewise. Good as-is.
    return CloudAugmentedDataset(
        raw_x, cont, raw_y, mean_x, std_x, mean_y, std_y,
        n_cloudy=N_CLOUDY, seed=SEED,
    )


def _build_val_dataset(cfg):
    """Clear validation set for in-loop monitoring, standardized with the model's
    own stats (augmented stats, or cache_planet stats for causal)."""
    base = os.path.join(DATA, cfg["base_cache"])
    aug = os.path.join(DATA, cfg["cloudaug"])
    stats = os.path.join(DATA, cfg["stats_cache"]) if cfg["stats_cache"] else aug

    val_x = torch.load(os.path.join(base, "val_x.pt"), map_location="cpu")
    val_y = torch.load(os.path.join(base, "val_y.pt"), map_location="cpu")
    mean_x = torch.load(os.path.join(stats, "mean_x.pt"), map_location="cpu")
    std_x = torch.load(os.path.join(stats, "std_x.pt"), map_location="cpu")
    mean_y = torch.load(os.path.join(stats, "mean_y.pt"), map_location="cpu")
    std_y = torch.load(os.path.join(stats, "std_y.pt"), map_location="cpu")

    val_x = (val_x - mean_x) / (std_x + 1e-30)
    val_y = (val_y - mean_y) / (std_y + 1e-8)
    return TensorDataset(val_x.float(), val_y.float())


def _build_counterfactual_dataset(cfg):
    """Causal track: load the pre-materialized env-counterfactual set (already
    standardized with cache_planet stats). Read-only."""
    cf_x = torch.load(cfg["cf_x"], map_location="cpu").float()
    cf_y = torch.load(cfg["cf_y"], map_location="cpu").float()
    print(f"  + counterfactual set {tuple(cf_x.shape)} concatenated")
    return TensorDataset(cf_x, cf_y)


def train_track(track: str, smoke: bool = False, num_workers: int = 0):
    if track not in TRACKS:
        raise SystemExit(f"Unknown track '{track}'. Choose from {list(TRACKS)}.")
    cfg = TRACKS[track]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    out_dir = os.path.join(THIS_DIR, track)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    print(f"\n=== Cloud-robust training: {track} on {device} ===")

    ModelClass = _load_model_class(cfg["model_py"], cfg["model_cls"])

    # Datasets
    train_ds = _build_cloud_dataset(cfg)
    if cfg["counterfactual"]:
        train_ds = ConcatDataset([train_ds, _build_counterfactual_dataset(cfg)])
    val_ds = _build_val_dataset(cfg)

    if smoke:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(2048, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(1024, len(val_ds)))))
        cfg = {**cfg, "epochs": 1}
        print("  [SMOKE] 1 epoch on a small subset")

    # Infer shape from one sample
    x0, _ = train_ds[0]
    in_channels, seq_len = x0.shape[0], x0.shape[1]
    print(f"  channels={in_channels} seq_len={seq_len} train={len(train_ds)} val={len(val_ds)}")
    model = ModelClass(in_channels=in_channels, sequence_length=seq_len).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                              pin_memory=(device.type == "cuda"), num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch"], shuffle=False,
                            num_workers=num_workers)

    criterion = nn.L1Loss() if cfg["loss"] == "l1" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = None
    if cfg["scheduler"] == "cosine_warmup":
        scheduler = _cosine_warmup(optimizer, cfg.get("warmup_epochs", 5),
                                   cfg["epochs"], len(train_loader))

    # Resume
    start_epoch, best_val, no_improve = 0, float("inf"), 0
    train_hist, val_hist = [], []
    resume = os.path.join(ckpt_dir, "checkpoint.pth")
    if os.path.isfile(resume) and not smoke:
        try:
            ck = torch.load(resume, map_location=device)
            model.load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["optimizer"])
            if scheduler is not None and ck.get("scheduler"):
                scheduler.load_state_dict(ck["scheduler"])
            start_epoch = ck["epoch"]
            best_val = ck["best_loss"]
            no_improve = ck.get("epochs_no_improve", 0)
            train_hist = ck.get("train_loss_history", [])
            val_hist = ck.get("val_loss_history", [])
            print(f"  resumed from epoch {start_epoch} (best_val={best_val:.6f})")
        except Exception as e:
            print(f"  could not resume ({e}); starting fresh")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        tl = 0.0
        for spectra, targets in tqdm(train_loader, desc=f"E{epoch+1} train"):
            spectra, targets = spectra.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(spectra)
            loss = criterion(pred, targets)
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            tl += loss.item()
        tl /= max(1, len(train_loader))
        train_hist.append(tl)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for spectra, targets in val_loader:
                spectra, targets = spectra.to(device), targets.to(device)
                vl += criterion(model(spectra), targets).item()
        vl /= max(1, len(val_loader))
        val_hist.append(vl)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch+1}/{cfg['epochs']}  train={tl:.6f}  val={vl:.6f}  lr={lr_now:.2e}")

        is_best = vl < best_val
        best_val = min(best_val, vl)
        no_improve = 0 if is_best else no_improve + 1

        if not smoke:
            state = {
                "epoch": epoch + 1, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_loss": best_val, "epochs_no_improve": no_improve,
                "train_loss_history": train_hist, "val_loss_history": val_hist,
                "track": track,
            }
            torch.save(state, os.path.join(ckpt_dir, "checkpoint.pth"))
            if is_best:
                torch.save(state, os.path.join(ckpt_dir, "model_best.pth"))

        if no_improve >= cfg["patience"]:
            print(f"  early stopping (no improve {no_improve} epochs)")
            break

    if train_hist and not smoke:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_hist) + 1), train_hist, marker="o", label="train")
        plt.plot(range(1, len(val_hist) + 1), val_hist, marker="o", label="val")
        plt.xlabel("epoch"); plt.ylabel(f"loss ({cfg['loss'].upper()})")
        plt.title(f"Cloud-robust {track}: loss curve"); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(charts_dir, "loss_curve.png"), dpi=150)
        plt.close()
    print(f"=== Done {track}. Best val loss = {best_val:.6f} ===")


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=list(TRACKS))
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a tiny subset")
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()
    train_track(a.track, smoke=a.smoke, num_workers=a.num_workers)


if __name__ == "__main__":
    _cli()
