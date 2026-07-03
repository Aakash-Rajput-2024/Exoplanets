"""Frozen, content-addressed train/val/test splits keyed by planet_index (C3/H2).

The split is a deterministic function of (the full sorted planet-id universe,
seed, ratios) — never of filesystem enumeration order — so it reproduces exactly
across machines and is *robust to incidental file skips*: a planet's partition is
decided from the complete summary.csv id list, so an unreadable spectrum drops one
sample rather than reshuffling everyone.

IDs are persisted so every downstream consumer (dataloaders, the causal
env-conditioning, grouped analyses) recovers identity by lookup instead of the
brittle KDTree-on-labels hack the causal track currently uses (H2).
"""

from __future__ import annotations

import json

import numpy as np

SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test


def make_splits(planet_ids, ratios=SPLIT_RATIOS, seed=SPLIT_SEED):
    """Partition the *full* planet-id universe into disjoint train/val/test.

    Parameters
    ----------
    planet_ids : iterable of str
        Every planet id in summary.csv (e.g. the zero-padded filename stems).
        Deduplicated and sorted internally, so input order is irrelevant.
    """
    ids = sorted({str(p) for p in planet_ids})
    if len(ids) < 3:
        raise ValueError(f"need >=3 planets to make a 3-way split, got {len(ids)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    idx = {
        "train": perm[:n_tr],
        "val": perm[n_tr:n_tr + n_va],
        "test": perm[n_tr + n_va:],
    }
    return {k: sorted(ids[i] for i in v) for k, v in idx.items()}


def split_lookup(splits):
    """Invert {split: [ids]} into {id: split} for O(1) per-file assignment."""
    return {pid: name for name, ids in splits.items() for pid in ids}


def save_splits(splits, path, meta=None):
    payload = {
        "seed": SPLIT_SEED,
        "ratios": list(SPLIT_RATIOS),
        "counts": {k: len(v) for k, v in splits.items()},
        **(meta or {}),
        "splits": splits,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_splits(path):
    with open(path) as f:
        return json.load(f)["splits"]
