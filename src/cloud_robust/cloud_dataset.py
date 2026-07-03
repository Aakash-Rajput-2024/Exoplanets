"""On-the-fly cloud-augmented dataset.

Wraps a base cache's RAW (un-standardized) `train_x` plus a precomputed
continuum tensor and labels. Length is 3*N:

    idx in [0,   N)  -> clear sample (f=0)
    idx in [N,  3N)  -> cloudy sample with fresh random (f, b) each access

So every epoch a planet appears once clear + twice cloudy ("clear + 2 cloudy
variants"), but the clouds are re-sampled every epoch -> stronger, free
augmentation with no 3x copy materialized on disk.

Inputs are standardized channel-wise with the AUGMENTED-distribution stats
(produced by build_cloud_cache.py) so cloudy inputs land on the same scale the
cloud-robust model is trained on. Labels are already standardized in the base
cache splits? No -- base caches store RAW labels; we standardize them here with
the augmented mean_y/std_y (identical to base label stats since clouds do not
change labels).

ISOLATION: reads only the cloud-aug cache; never writes anything.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cloud_model import apply_cloud, sample_cloud_params, cloud_channel_for_mode, F_RANGE, B_RANGE


class CloudAugmentedDataset(Dataset):
    def __init__(
        self,
        raw_x: torch.Tensor,        # (N, C, L) RAW signals
        continuum: torch.Tensor,    # (N, C, L) precomputed continuum (same units)
        raw_y: torch.Tensor,        # (N, 12) RAW labels
        mean_x: torch.Tensor,       # (1, C, 1) augmented-distribution input stats
        std_x: torch.Tensor,
        mean_y: torch.Tensor,       # (1, 12) label stats
        std_y: torch.Tensor,
        n_cloudy: int = 2,
        f_range=F_RANGE,
        b_range=B_RANGE,
        seed: int = 0,
    ):
        assert raw_x.shape == continuum.shape, "continuum must match raw_x shape"
        self.raw_x = raw_x
        self.continuum = continuum
        self.N = raw_x.shape[0]
        self.C = raw_x.shape[1]
        self.cloud_ch = cloud_channel_for_mode(self.C)
        self.n_cloudy = n_cloudy
        self.f_range = f_range
        self.b_range = b_range
        self.seed = seed

        # Standardized labels (clouds do not change labels, so repeat per copy).
        self.y_std = ((raw_y - mean_y) / (std_y + 1e-8)).float()

        self.mean_x = mean_x.float()
        self.std_x = std_x.float()

    def __len__(self):
        return self.N * (1 + self.n_cloudy)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_x.squeeze(0)) / (self.std_x.squeeze(0) + 1e-30)

    def __getitem__(self, idx):
        planet = idx % self.N
        copy_id = idx // self.N  # 0 = clear, >=1 = cloudy
        y = self.y_std[planet]

        x = self.raw_x[planet].clone()  # (C, L)
        if copy_id == 0:
            return self._standardize(x), y

        # Fresh per-access cloud draw. Seed couples (epoch-free) to planet+copy so
        # different copies differ but a given access is reproducible within a run.
        rng = np.random.default_rng((self.seed, planet, copy_id, torch.randint(0, 2**31 - 1, (1,)).item()))
        f, b = sample_cloud_params(rng, self.f_range, self.b_range)

        ch = self.cloud_ch
        p = x[ch].numpy()
        c = self.continuum[planet, ch].numpy()
        x[ch] = torch.from_numpy(apply_cloud(p, f, b, continuum=c).astype(np.float32))
        return self._standardize(x), y


def make_cloudy_val(
    raw_val_x: torch.Tensor,
    continuum_val: torch.Tensor,
    f: float,
    b: float,
) -> torch.Tensor:
    """Materialize a fixed cloudy validation set at a single (f, b).

    Used by build_cloud_cache.py to write reproducible eval sets and the opacity
    sweep. Operates on the cloud channel only (planet_signal).
    """
    C = raw_val_x.shape[1]
    ch = cloud_channel_for_mode(C)
    out = raw_val_x.clone()
    p = raw_val_x[:, ch, :].numpy()
    c = continuum_val[:, ch, :].numpy()
    out[:, ch, :] = torch.from_numpy(apply_cloud(p, f, b, continuum=c).astype(np.float32))
    return out
