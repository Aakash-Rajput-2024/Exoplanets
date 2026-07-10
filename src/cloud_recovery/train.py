"""Train the declouder (supervised restoration).  ISOLATED — writes only decloud/.

    PYTHONPATH=src python -m decloud.train --seed 0
    PYTHONPATH=src python -m decloud.train --smoke          # 1 epoch on a subset

Learns  encoded clouded observable [B,2,L]  →  encoded clear contrast [B,1,L]
with a Huber loss in the same asinh-normed space the retrieval models train in.
Grey clouds (+ exposure noise) are drawn fresh per batch; the held-out families are
kept for the eval so generalisation is measured, never fitted.

Nothing here trains or touches a retrieval model, cache, or checkpoint — it reads
``data/cache_v2`` read-only and writes ``src/cloud_recovery/checkpoints/``.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.registry import CACHE_V2
from common.runtime import get_device
from decloud.cloud_pairs import (CloudPairDataset, GreyCloudPairCollate, build_fixed_pairs,
                                 get_norm_v2, DEFAULT_ALPHA_RANGE)
from decloud.model import DecloudUNet1D, count_params

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(THIS_DIR, "checkpoints")


def _cosine_warmup(optimizer, warmup, total):
    def fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        prog = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def _mean_loss(model, inp, tgt, device, criterion, batch=256):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, inp.shape[0], batch):
        xb = inp[i:i + batch].to(device)
        yb = tgt[i:i + batch].to(device)
        tot += criterion(model(xb), yb).item() * xb.shape[0]
        n += xb.shape[0]
    return tot / max(1, n)


@torch.no_grad()
def _identity_loss(inp, tgt, criterion, batch=256):
    """Huber(clouded ch0, clear ch0) — the score of doing nothing (the residual's
    starting point). The gap between this and the model's val loss is the win."""
    tot, n = 0.0, 0
    for i in range(0, inp.shape[0], batch):
        xb = inp[i:i + batch, 0:1, :]
        yb = tgt[i:i + batch]
        tot += criterion(xb, yb).item() * xb.shape[0]
        n += xb.shape[0]
    return tot / max(1, n)


def train(seed=0, epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4, warmup=5,
          patience=15, num_workers=0, base=32, depth=3, val_family="grey", val_alpha=30.0,
          val_n=4000, smoke=False, suffix="", cache_dir=CACHE_V2):
    device = get_device()
    torch.manual_seed(seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    norm = get_norm_v2(cache_dir)

    train_ds = CloudPairDataset(cache_dir, "train", max_rows=(2048 if smoke else None))
    if smoke:
        epochs, val_n = 1, 512
    collate = GreyCloudPairCollate(norm, base_seed=seed, alpha_range=DEFAULT_ALPHA_RANGE)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=collate, drop_last=True)

    # Fixed val pairs (deterministic) → a stable early-stopping signal.
    val_inp, val_tgt = build_fixed_pairs(cache_dir, "val", norm, family=val_family,
                                         alpha=val_alpha, n_max=val_n)
    L = val_inp.shape[-1]

    model = DecloudUNet1D(in_channels=val_inp.shape[1], sequence_length=L, base=base, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _cosine_warmup(opt, warmup, epochs)
    criterion = nn.SmoothL1Loss(beta=1.0)      # Huber δ=1, matches the retrieval budget

    id_val = _identity_loss(val_inp, val_tgt, criterion)
    print(f"decloud train | seed {seed} | {device} | {count_params(model):,} params | "
          f"L={L} | val identity(Huber)={id_val:.5f}")

    best_val, best_epoch, no_improve, history = float("inf"), -1, 0, []
    for epoch in range(epochs):
        if hasattr(collate, "set_epoch"):
            collate.set_epoch(epoch)
        model.train()
        run, nb = 0.0, 0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad()
            loss = criterion(model(inp), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item(); nb += 1
        sched.step()
        tr = run / max(1, nb)
        vl = _mean_loss(model, val_inp, val_tgt, device, criterion)
        is_best = vl < best_val
        rel = 100.0 * (1.0 - vl / id_val) if id_val > 0 else float("nan")
        print(f"  epoch {epoch + 1:>3}/{epochs} | lr {sched.get_last_lr()[0]:.2e} | "
              f"train {tr:.5f} | val {vl:.5f} | vs-identity {rel:+.1f}%"
              f"{'  *best' if is_best else ''}")
        history.append(dict(epoch=epoch + 1, train=tr, val=vl, rel_improve_pct=rel))
        if is_best:
            best_val, best_epoch, no_improve = vl, epoch + 1, 0
            _save(model, seed, suffix, epoch + 1, best_val, id_val, history, L, base, depth)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop (no val improvement for {patience} epochs)")
                break

    print(f"=> done. best val Huber {best_val:.5f} @epoch {best_epoch} "
          f"({100.0 * (1 - best_val / id_val):+.1f}% vs identity); "
          f"ckpt: {_ckpt_path(seed, suffix)}")
    return best_val


def _ckpt_path(seed, suffix=""):
    return os.path.join(CKPT_DIR, f"decloud{suffix}_seed{seed}.pth")


def _save(model, seed, suffix, epoch, best_val, id_val, history, seq_len, base, depth):
    torch.save({
        "state_dict": model.state_dict(),
        "in_channels": model.in_channels,
        "seq_len": seq_len,
        "epoch": epoch,
        "best_val": best_val,
        "val_identity": id_val,
        "history": history,
        "config": dict(base=base, depth=depth, residual=True, obs_mode="contrast_snr",
                       input_norm="per_lambda_asinh", train_family="grey",
                       alpha_range=DEFAULT_ALPHA_RANGE, seed=seed),
    }, _ckpt_path(seed, suffix))


def main():
    ap = argparse.ArgumentParser(description="Train the isolated spectral declouder.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--base", type=int, default=32, help="U-Net first-stage width")
    ap.add_argument("--depth", type=int, default=3, help="U-Net down/up stages")
    ap.add_argument("--val-family", default="grey", help="cloud family for the val metric")
    ap.add_argument("--val-alpha", type=float, default=30.0)
    ap.add_argument("--val-n", type=int, default=4000, help="max val planets (speed)")
    ap.add_argument("--suffix", default="", help="checkpoint name suffix")
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a small subset")
    a = ap.parse_args()
    train(seed=a.seed, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          weight_decay=a.weight_decay, warmup=a.warmup, patience=a.patience,
          num_workers=a.num_workers, base=a.base, depth=a.depth, val_family=a.val_family,
          val_alpha=a.val_alpha, val_n=a.val_n, smoke=a.smoke, suffix=a.suffix, cache_dir=a.cache)


if __name__ == "__main__":
    main()
