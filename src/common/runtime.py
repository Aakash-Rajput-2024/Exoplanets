"""Determinism + device selection shared by every track (fixes H6, M4).

The review flagged that (a) no track seeds torch/numpy/python, so single-run
0.70-vs-0.72 gaps were read as findings (H6), and (b) `test.py` selected only
cuda-else-cpu while `train.py` used MPS, giving inconsistent numerics between
train and eval (M4). Both are one-liners once centralized here.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (all devices) and prefer deterministic kernels.

    Determinism on MPS/CUDA is best-effort — some conv/pool kernels have no
    deterministic implementation — but the RNG streams (init, dropout, shuffle,
    noise injection) are fully reproducible, which is what the multi-seed study
    (H6) needs to separate architecture effects from initialization noise.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_device() -> torch.device:
    """cuda → mps → cpu, used identically in training and evaluation (M4)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
