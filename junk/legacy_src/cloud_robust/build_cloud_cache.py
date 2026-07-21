"""Build cloud-augmentation caches from existing INARA caches (read-only).

For each base cache, writes a sibling `data/cache_cloudaug_<name>/` containing
everything the cloud-robust training/eval needs that the base cache does NOT
already provide:

    continuum_train.pt   (N,   C, L)  precomputed band-free continuum for train
    val_continuum.pt     (Nv,  C, L)  continuum for val (eval generates cloudy
                                      on the fly from base val_x + this)
    mean_x.pt, std_x.pt               channel-wise input stats over the AUGMENTED
                                      (clear + n_cloudy cloudy) training distribution
    mean_y.pt, std_y.pt               label stats (clouds do not change labels)
    manifest.json                     base cache path, channels, params

The big raw tensors (train_x/train_y/val_x/val_y) are NOT copied -- training and
eval read those from the base cache directly. Only the continuum (one tensor the
size of train_x) is materialized.

ISOLATION: base caches are opened read-only; nothing under data/cache,
data/cache_original, data/cache_planet is ever written.

USAGE
    python build_cloud_cache.py            # builds all three canonical caches
    python build_cloud_cache.py original   # build just one (original|planet|both)
"""

from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cloud_model import (
    estimate_continuum, apply_cloud, cloud_channel_for_mode, F_RANGE, B_RANGE,
)

DATA_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/data"

# base cache name -> output cache name. Each base preserves its own 80/20 split.
CANONICAL = {
    "original": ("cache_original", "cache_cloudaug_original"),  # 1ch -> original1dcnn
    "planet":   ("cache_planet",   "cache_cloudaug_planet"),    # 1ch -> transformer, causal
    "both":     ("cache",          "cache_cloudaug_both"),       # 2ch -> optimized, 2channel
}

N_CLOUDY = 2          # cloudy variants per planet used to estimate aug stats
CHUNK = 4000          # rows per chunk (memory control)
SEED = 12345


def _continuum_chunked(raw_x: torch.Tensor) -> torch.Tensor:
    """Continuum per channel, computed in row-chunks to bound memory."""
    N, C, L = raw_x.shape
    out = torch.empty((N, C, L), dtype=torch.float32)
    for lo in range(0, N, CHUNK):
        hi = min(N, lo + CHUNK)
        chunk = raw_x[lo:hi].numpy().astype(np.float64)        # (n, C, L)
        cont = estimate_continuum(chunk)                       # over last axis
        out[lo:hi] = torch.from_numpy(cont.astype(np.float32))
        print(f"    continuum {hi}/{N}", end="\r")
    print()
    return out


def _augmented_input_stats(raw_x: torch.Tensor, cont: torch.Tensor, n_cloudy: int):
    """Channel-wise mean/std over the augmented (clear + n_cloudy cloudy) set.

    Matches the base dataloader convention: stats over dims (0, 2), keepdim.
    Accumulates sum / sumsq in chunks; cloudy draws use per-row (f, b).
    """
    N, C, L = raw_x.shape
    ch = cloud_channel_for_mode(C)
    rng = np.random.default_rng(SEED)

    ssum = np.zeros(C, dtype=np.float64)
    ssq = np.zeros(C, dtype=np.float64)
    count = 0

    def accumulate(arr):  # arr: (n, C, L) float64
        nonlocal ssum, ssq, count
        ssum += arr.sum(axis=(0, 2))
        ssq += (arr ** 2).sum(axis=(0, 2))
        count += arr.shape[0] * L

    for lo in range(0, N, CHUNK):
        hi = min(N, lo + CHUNK)
        x = raw_x[lo:hi].numpy().astype(np.float64)            # (n, C, L)
        c = cont[lo:hi].numpy().astype(np.float64)
        accumulate(x)                                          # clear copy
        for _ in range(n_cloudy):
            n = hi - lo
            f = rng.uniform(*F_RANGE, size=(n, 1))             # per-row params
            b = rng.uniform(*B_RANGE, size=(n, 1))
            xc = x.copy()
            xc[:, ch, :] = apply_cloud(x[:, ch, :], f, b, continuum=c[:, ch, :])
            accumulate(xc)
        print(f"    stats {hi}/{N}", end="\r")
    print()

    mean = ssum / count
    var = np.clip(ssq / count - mean ** 2, a_min=0.0, a_max=None)
    std = np.sqrt(var)
    mean_x = torch.tensor(mean, dtype=torch.float32).view(1, C, 1)
    std_x = torch.tensor(std, dtype=torch.float32).view(1, C, 1)
    return mean_x, std_x


def build_one(key: str):
    base_name, out_name = CANONICAL[key]
    base = os.path.join(DATA_DIR, base_name)
    out = os.path.join(DATA_DIR, out_name)
    os.makedirs(out, exist_ok=True)

    print(f"\n=== Building {out_name} from {base_name} (read-only) ===")
    raw_x = torch.load(os.path.join(base, "train_x.pt"), map_location="cpu")
    raw_y = torch.load(os.path.join(base, "train_y.pt"), map_location="cpu")
    val_x = torch.load(os.path.join(base, "val_x.pt"), map_location="cpu")
    C = raw_x.shape[1]
    print(f"  train_x {tuple(raw_x.shape)}  val_x {tuple(val_x.shape)}  channels={C}  cloud_ch={cloud_channel_for_mode(C)}")

    print("  [1/4] train continuum ...")
    cont_train = _continuum_chunked(raw_x)
    torch.save(cont_train, os.path.join(out, "continuum_train.pt"))

    print("  [2/4] val continuum ...")
    cont_val = _continuum_chunked(val_x)
    torch.save(cont_val, os.path.join(out, "val_continuum.pt"))

    print("  [3/4] augmented input stats ...")
    mean_x, std_x = _augmented_input_stats(raw_x, cont_train, N_CLOUDY)
    torch.save(mean_x, os.path.join(out, "mean_x.pt"))
    torch.save(std_x, os.path.join(out, "std_x.pt"))

    print("  [4/4] label stats ...")
    mean_y = raw_y.mean(dim=0, keepdim=True)
    std_y = raw_y.std(dim=0, keepdim=True)
    torch.save(mean_y, os.path.join(out, "mean_y.pt"))
    torch.save(std_y, os.path.join(out, "std_y.pt"))

    manifest = {
        "base_cache": base,
        "out_cache": out,
        "channels": int(C),
        "cloud_channel": int(cloud_channel_for_mode(C)),
        "n_cloudy_for_stats": N_CLOUDY,
        "f_range": list(F_RANGE),
        "b_range": list(B_RANGE),
        "seed": SEED,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Training reads raw train_x/train_y from base_cache; this dir holds "
                "continuum + augmented stats only. Eval generates cloudy val on the "
                "fly from base val_x + val_continuum.",
    }
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Sanity assertions.
    assert torch.isfinite(mean_x).all() and torch.isfinite(std_x).all(), "non-finite input stats"
    assert (std_x >= 0).all(), "negative std"
    assert cont_train.shape == raw_x.shape and cont_val.shape == val_x.shape
    print(f"  mean_x={mean_x.flatten().tolist()}  std_x={std_x.flatten().tolist()}")
    print(f"=> Done: {out}")


def main():
    keys = sys.argv[1:] or list(CANONICAL.keys())
    for k in keys:
        if k not in CANONICAL:
            raise SystemExit(f"Unknown cache key '{k}'. Choose from {list(CANONICAL)}.")
        build_one(k)


if __name__ == "__main__":
    main()
