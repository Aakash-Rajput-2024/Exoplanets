"""Build the unified v2 cache (fixes C3/H2, foundation for C1/C2/H3/C4).

ONE cache serves every track. Per split it stores the RAW (un-transformed)
observables plus identity + metadata, so the label space (C2), the input
encoding (H3) and noise injection (C1) are all *load-time* choices — the 94 GB
of CSVs is parsed exactly once.

Layout (data/cache_v2/ by default):
    wavelength.pt              (L,)       shared λ grid, saved once
    {split}_x.pt               (N,2,L)    ch0=star_planet_signal, ch1=planet_signal (raw erg/s/cm2)
    {split}_noise.pt           (N,L)      instrument noise column (raw erg/s/cm2)
    {split}_y.pt               (N,12)     LINEAR mole fractions, TARGET order
    {split}_ids.json           [str]      planet_index per row, in tensor order
    {split}_meta.csv                      full summary rows for the split, in tensor order
    manifest.json                         schema, seed/ratios, counts, skip log, provenance

Design guarantees:
  * Split is decided from the COMPLETE summary id universe (splits.make_splits),
    so it is reproducible and robust to incidental file skips.
  * NO silent skips: every skipped file is logged with a reason; the build aborts
    if the skip fraction exceeds --max-skip-frac.
  * λ grid is verified identical for every accepted file.

Memory: accumulates the full dataset in RAM (~6-8 GB of float32 tensors). Run on
a machine with headroom, or use --limit for a smoke build.

    python -m common.build_cache                 # full build -> data/cache_v2
    python -m common.build_cache --limit 300 --out data/cache_v2_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from common.splits import make_splits, split_lookup, save_splits, SPLIT_SEED, SPLIT_RATIOS
from common.provenance import git_commit, config_hash

TARGET_COLUMNS = [
    "H2O", "CO2", "O2", "N2", "CH4", "N2O",
    "CO", "O3", "SO2", "NH3", "C2H6", "NO2",
]
# Input channel order stored in {split}_x.pt (feature_mode selects at load time).
CHANNELS = ["star_planet_signal_(erg/s/cm2)", "planet_signal_(erg/s/cm2)"]
NOISE_COL = "noise_(erg/s/cm2)"
WL_COL = "wavelength_(um)"
WL_MAX_UM = 2.0


def _read_spectrum(path, wl_ref):
    """Return (x[2,L], noise[L], wl[L]) for one CSV, or raise with a reason."""
    df = pd.read_csv(path)
    df = df[df[WL_COL] <= WL_MAX_UM]
    wl = df[WL_COL].to_numpy(dtype=np.float64)
    if wl_ref is not None:
        if len(wl) != len(wl_ref) or not np.allclose(wl, wl_ref, rtol=0, atol=1e-9):
            raise ValueError(f"grid_mismatch (len={len(wl)})")
    chans = []
    for c in CHANNELS:
        v = np.nan_to_num(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64))
        chans.append(v)
    noise = np.nan_to_num(pd.to_numeric(df[NOISE_COL], errors="coerce").to_numpy(dtype=np.float64))
    x = np.stack(chans, axis=0).astype(np.float32)      # (2, L)
    return x, noise.astype(np.float32), wl


def build(spectra_dir, summary_path, out_dir, limit=None, max_skip_frac=0.005):
    os.makedirs(out_dir, exist_ok=True)
    summary = pd.read_csv(summary_path, dtype={"planet_index": str}).set_index("planet_index")
    missing_cols = [c for c in TARGET_COLUMNS if c not in summary.columns]
    if missing_cols:
        raise ValueError(f"summary.csv missing target columns: {missing_cols}")

    # Frozen split over the COMPLETE id universe (robust to skips).
    splits = make_splits(summary.index.tolist())
    assign = split_lookup(splits)

    files = sorted(f for f in os.listdir(spectra_dir) if f.endswith(".csv"))
    if limit:
        files = files[:limit]
    print(f"[build] {len(files)} spectra -> {out_dir}  (limit={limit})")

    acc = {s: {"x": [], "noise": [], "y": [], "ids": []} for s in splits}
    skipped = []          # (filename, reason)
    wl_ref = None
    n_ok = 0

    for i, fn in enumerate(files):
        pid = fn.split(".")[0]                     # zero-padded stem == planet_index
        if pid not in assign:
            skipped.append((fn, "not_in_split_universe"))
            continue
        if pid not in summary.index:
            skipped.append((fn, "no_summary_row"))
            continue
        try:
            x, noise, wl = _read_spectrum(os.path.join(spectra_dir, fn), wl_ref)
        except Exception as e:
            skipped.append((fn, f"read_error:{type(e).__name__}:{e}"))
            continue
        if wl_ref is None:
            wl_ref = wl
        y = pd.to_numeric(summary.loc[pid, TARGET_COLUMNS], errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(y).any():
            skipped.append((fn, "nan_label"))
            continue
        s = assign[pid]
        acc[s]["x"].append(x)
        acc[s]["noise"].append(noise)
        acc[s]["y"].append(y)
        acc[s]["ids"].append(pid)
        n_ok += 1
        if (i + 1) % 10000 == 0:
            print(f"  processed {i + 1}/{len(files)}  (kept {n_ok}, skipped {len(skipped)})")

    skip_frac = len(skipped) / max(1, len(files))
    print(f"[build] kept {n_ok}, skipped {len(skipped)} ({skip_frac:.4%})")
    if skip_frac > max_skip_frac:
        reasons = {}
        for _, r in skipped:
            reasons[r.split(":")[0]] = reasons.get(r.split(":")[0], 0) + 1
        raise RuntimeError(
            f"skip fraction {skip_frac:.4%} exceeds --max-skip-frac {max_skip_frac:.4%}; "
            f"reasons={reasons}. Refusing to write a silently-incomplete cache."
        )

    # Save per-split tensors + identity + metadata, all in the same row order.
    torch.save(torch.from_numpy(np.asarray(wl_ref, dtype=np.float32)), os.path.join(out_dir, "wavelength.pt"))
    kept_ids = {}
    for s in splits:
        ids = acc[s]["ids"]
        kept_ids[s] = ids
        if not ids:
            print(f"  [warn] split '{s}' is empty (limit too small?)")
            continue
        torch.save(torch.from_numpy(np.stack(acc[s]["x"])), os.path.join(out_dir, f"{s}_x.pt"))
        torch.save(torch.from_numpy(np.stack(acc[s]["noise"])), os.path.join(out_dir, f"{s}_noise.pt"))
        torch.save(torch.from_numpy(np.stack(acc[s]["y"])), os.path.join(out_dir, f"{s}_y.pt"))
        with open(os.path.join(out_dir, f"{s}_ids.json"), "w") as f:
            json.dump(ids, f)
        summary.loc[ids].to_csv(os.path.join(out_dir, f"{s}_meta.csv"))
        print(f"  wrote {s}: N={len(ids)}")

    build_cfg = dict(
        spectra_dir=os.path.relpath(spectra_dir, REPO_ROOT),
        summary=os.path.relpath(summary_path, REPO_ROOT),
        seed=SPLIT_SEED, ratios=list(SPLIT_RATIOS), wl_max_um=WL_MAX_UM,
        channels=CHANNELS, targets=TARGET_COLUMNS, limit=limit,
    )
    manifest = dict(
        schema_version=2,
        **build_cfg,
        L=int(len(wl_ref)) if wl_ref is not None else None,
        counts={s: len(kept_ids[s]) for s in splits},
        n_total_files=len(files), n_kept=n_ok, n_skipped=len(skipped),
        skip_fraction=skip_frac, skipped=skipped[:1000],  # cap log size
        git_commit=git_commit(REPO_ROOT), config_hash=config_hash(build_cfg),
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    # Persist the split id lists standalone too (consumed by loaders / causal track).
    save_splits(splits, os.path.join(out_dir, "splits.json"),
                meta={"note": "split over full summary id universe; kept ids are per-split *_ids.json"})
    print(f"[build] done. manifest -> {os.path.join(out_dir, 'manifest.json')}")
    return manifest


def _cli():
    ap = argparse.ArgumentParser(description="Build the unified v2 INARA cache.")
    ap.add_argument("--spectra-dir", default=os.path.join(REPO_ROOT, "data", "inara_1by3"))
    ap.add_argument("--summary", default=os.path.join(REPO_ROOT, "data", "summary.csv"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data", "cache_v2"))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (smoke build)")
    ap.add_argument("--max-skip-frac", type=float, default=0.005)
    a = ap.parse_args()
    build(a.spectra_dir, a.summary, a.out, limit=a.limit, max_skip_frac=a.max_skip_frac)


if __name__ == "__main__":
    _cli()
