"""Per-wavelength input encoding + noise-injecting datasets (fixes H3, serves C1).

INPUT ENCODING (H3)
    The old pipeline standardized each channel with a SINGLE scalar mean/std over
    all 4379 wavelengths, so the blue end (fluxes ~1e-28) collapsed into a
    numerically constant band and the first-layer tanh saturated on the bright
    end. We instead use a per-λ **asinh** normalization:

        t(λ) = asinh( x(λ) / s(λ) ) ,  then  (t - μ(λ)) / σ(λ)

    with s, μ, σ vectors of length L fitted on TRAIN. asinh is linear through
    zero and logarithmic in the tails, so it compresses the 16-decade contrast
    range AND tolerates the negative values that noise injection produces (a plain
    log10 cannot). Per-λ μ/σ give every wavelength unit variance, so faint
    short-λ ozone/Rayleigh structure is no longer invisible.

NOISE (C1)
    ``NoisyObservableDataset`` injects fresh, *seeded* noise each epoch
    (set_epoch) — reproducible, unlike the mixed seeded/unseeded draw the review
    flagged (H6). ``build_eval_observable`` materializes a fixed noisy set at a
    given SNR multiplier α for the sweep.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from common.observable import make_observable, inject_noise

_STD_EPS = 1e-8


class PerLambdaAsinhNorm:
    """asinh softening + per-wavelength standardization, fitted on TRAIN."""

    def __init__(self, scale, mean, std):
        self.scale = scale      # (1, C, L)
        self.mean = mean        # (1, C, L)
        self.std = std          # (1, C, L)

    @classmethod
    def fit(cls, x_train):
        # Per-λ softening = median|x| (robust typical scale), floored away from 0.
        scale = x_train.abs().median(dim=0, keepdim=True).values.clamp(min=1e-20)
        t = torch.asinh(x_train / scale)
        return cls(scale, t.mean(dim=0, keepdim=True), t.std(dim=0, keepdim=True))

    def encode(self, x):
        t = torch.asinh(x / self.scale)
        return (t - self.mean) / (self.std + _STD_EPS)

    def state(self):
        return {"scale": self.scale, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s):
        return cls(s["scale"], s["mean"], s["std"])


def build_clean_observable(raw_x, noise, obs_mode, norm):
    """Noise-free encoded observable (val/test baseline)."""
    return norm.encode(make_observable(raw_x, noise, obs_mode))


def build_eval_observable(raw_x, noise, obs_mode, norm, alpha=1.0, seed=0):
    """Fixed noisy encoded observable at SNR multiplier α (for the eval sweep)."""
    gen = torch.Generator().manual_seed(int(seed))
    return norm.encode(inject_noise(raw_x, noise, obs_mode, alpha=alpha, generator=gen))


class NoisyObservableDataset(Dataset):
    """Training set that forms the observable and injects fresh seeded noise.

    Labels are pre-encoded (LabelPipeline). Call ``set_epoch(e)`` each epoch so
    the noise realization varies yet stays reproducible.
    """

    def __init__(self, raw_x, noise, y_encoded, obs_mode, norm, alpha=1.0, base_seed=0):
        self.raw_x = raw_x
        self.noise = noise
        self.y = y_encoded
        self.obs_mode = obs_mode
        self.norm = norm
        self.alpha = alpha
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.raw_x.shape[0]

    def __getitem__(self, i):
        seed = (self.base_seed * 1_000_003 + self.epoch * 9973 + i) % (2 ** 31 - 1)
        gen = torch.Generator().manual_seed(seed)
        obs = inject_noise(self.raw_x[i:i + 1], self.noise[i:i + 1],
                           self.obs_mode, alpha=self.alpha, generator=gen)
        return self.norm.encode(obs)[0], self.y[i]
