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

import math

import torch
from torch.utils.data import Dataset

from common.observable import make_observable, inject_noise

_STD_EPS = 1e-8


def _draw_log_uniform_alpha(n, alpha_range, generator):
    """Per-sample SNR multiplier α ~ log-uniform[lo, hi], shape (n, 1).

    Exposure-time augmentation: since α = √(t/t_nom), a log-uniform draw is
    uniform in log-exposure. Training over the whole eval-sweep range makes ONE
    model span every exposure instead of overfitting to a single noise level —
    without it the model specializes to the training α and the R²-vs-exposure
    sweep humps then DECLINES at high SNR (a covariate-shift artifact, not
    physics). See notes/ULTRAPLAN_review.md Part 3.
    """
    lo, hi = alpha_range
    u = torch.rand(n, 1, generator=generator)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


class PerLambdaAsinhNorm:
    """asinh softening + per-wavelength standardization, fitted on TRAIN."""

    kind = "per_lambda_asinh"

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
        return {"kind": self.kind, "scale": self.scale, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s):
        return cls(s["scale"], s["mean"], s["std"])


class ScalarNorm:
    """The OLD encoding: a single scalar mean/std PER CHANNEL over all λ.

    Kept only as the control arm of the input-representation ablation (paper
    experiment #10 / H3): running the exact same observable and labels through
    scalar-std vs PerLambdaAsinhNorm isolates the encoding's contribution. On its
    own this collapses the 0.2-0.5 µm band (fluxes ~1e-28) into a numerically
    constant strip — the failure H3 describes — so it is never the default.
    """

    kind = "scalar"

    def __init__(self, mean, std):
        self.mean = mean        # (1, C, 1)
        self.std = std          # (1, C, 1)

    @classmethod
    def fit(cls, x_train):
        return cls(x_train.mean(dim=(0, 2), keepdim=True),
                   x_train.std(dim=(0, 2), keepdim=True))

    def encode(self, x):
        return (x - self.mean) / (self.std + _STD_EPS)

    def state(self):
        return {"kind": self.kind, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s):
        return cls(s["mean"], s["std"])


_NORMS = {"per_lambda_asinh": PerLambdaAsinhNorm, "scalar": ScalarNorm}


def norm_from_state(s):
    """Rebuild whichever normalizer produced ``s`` (dispatch on its ``kind``)."""
    return _NORMS[s["kind"]].from_state(s)


def fit_norm(input_norm, x_train):
    """Fit the named normalizer on the (clean) train observable."""
    if input_norm not in _NORMS:
        raise ValueError(f"input_norm must be one of {list(_NORMS)}, got {input_norm!r}")
    return _NORMS[input_norm].fit(x_train)


def build_clean_observable(raw_x, noise, obs_mode, norm):
    """Noise-free encoded observable (val/test baseline)."""
    return norm.encode(make_observable(raw_x, noise, obs_mode))


def build_eval_observable(raw_x, noise, obs_mode, norm, alpha=1.0, seed=0, alpha_range=None):
    """Fixed noisy encoded observable at SNR multiplier α (for the eval sweep).

    ``alpha_range`` (lo, hi) instead draws a per-sample log-uniform α at this
    fixed seed — used for the VALIDATION set so model selection targets
    average-over-exposure performance (matching the augmented training
    distribution), not the near-zero α=1 photon floor. The test SNR sweep leaves
    it None: each sweep point measures a single, fixed exposure.
    """
    gen = torch.Generator().manual_seed(int(seed))
    a = _draw_log_uniform_alpha(raw_x.shape[0], alpha_range, gen) if alpha_range else alpha
    return norm.encode(inject_noise(raw_x, noise, obs_mode, alpha=a, generator=gen))


# --- Fast training path: raw datasets + vectorized batch collates -------------
# The observable/noise/asinh math is O(B·L) but done ONCE per batch (vectorized
# torch) instead of once per sample in __getitem__. Profiling showed the per-item
# version spent ~0.7 s/batch entirely in Python (a torch.Generator per sample +
# per-sample asinh on a 2×4379 slice), dwarfing the millisecond MPS forward pass.
# The dataset now only returns cheap tensor views; the collate does the work.


class RawObservableDataset(Dataset):
    """Returns raw (star_planet+planet, noise, label) tuples — no per-item math.

    Pair with :class:`NoiseCollate` (or a Subset thereof); the collate forms the
    observable and injects noise for the whole batch at once.
    """

    def __init__(self, raw_x, noise, y_encoded):
        self.raw_x = raw_x
        self.noise = noise
        self.y = y_encoded

    def __len__(self):
        return self.raw_x.shape[0]

    def __getitem__(self, i):
        return i, self.raw_x[i], self.noise[i], self.y[i]


class RawCloudObservableDataset(RawObservableDataset):
    """Like :class:`RawObservableDataset` but also carries the planet-channel
    continuum, so :class:`CloudNoiseCollate` can grey-cloud the batch (C5)."""

    def __init__(self, raw_x, noise, continuum_planet, y_encoded):
        super().__init__(raw_x, noise, y_encoded)
        self.continuum = continuum_planet

    def __getitem__(self, i):
        return i, self.raw_x[i], self.noise[i], self.continuum[i], self.y[i]


class NoiseCollate:
    """Vectorized collate: stack a raw batch, inject fresh seeded noise, encode.

    The noise seed = f(base_seed, epoch, batch indices), i.e. it is derived from
    the batch CONTENT rather than a mutable counter. That keeps a run fully
    reproducible AND multiprocessing-safe (each worker seeds identically for a
    given batch), so ``num_workers>0`` is allowed. Call ``set_epoch(e)`` before
    each epoch's iterator is created so worker copies inherit the epoch. Only the
    training stream depends on the model seed (val/test use fixed constants).
    """

    def __init__(self, obs_mode, norm, alpha=1.0, base_seed=0, alpha_range=None):
        self.obs_mode = obs_mode
        self.norm = norm
        self.alpha = alpha
        self.alpha_range = alpha_range   # (lo,hi) → per-sample exposure augmentation
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _gen(self, idxs):
        key = self.base_seed * 1_000_003 + self.epoch * 9973 + int(sum(idxs))
        return torch.Generator().manual_seed(key % (2 ** 31 - 1))

    def _alpha(self, n, gen):
        """Fixed α, unless ``alpha_range`` is set → per-sample log-uniform draw.
        When None, no RNG is consumed, so the non-augmented path is bit-identical
        to before (back-compat)."""
        if self.alpha_range is None:
            return self.alpha
        return _draw_log_uniform_alpha(n, self.alpha_range, gen)

    def __call__(self, batch):
        idxs = [b[0] for b in batch]
        xs = torch.stack([b[1] for b in batch])      # (B, 2, L)
        ns = torch.stack([b[2] for b in batch])      # (B, L)
        ys = torch.stack([b[3] for b in batch])      # (B, D)
        gen = self._gen(idxs)
        a = self._alpha(xs.shape[0], gen)
        obs = inject_noise(xs, ns, self.obs_mode, alpha=a, generator=gen)
        return self.norm.encode(obs), ys


class CloudNoiseCollate(NoiseCollate):
    """NoiseCollate + a per-sample grey cloud on the planet channel each epoch
    (the C5 training family). Draws (f, b) ~ U(f_range)×U(b_range) per sample from
    the same seeded stream, so f≈0 is clear and f→1 fully muted+brightened.

    The cloud modifies F_p, so the star+planet channel is rebuilt from the SAME
    cloudy planet (star_planet = F_star + F_p_cloudy). Otherwise ch0 would keep
    the CLEAR planet while ch1 is cloudy, making the contrast and the star-only
    SNR mutually inconsistent (and re-leaking the clear F_p through star_planet).
    """

    def __init__(self, obs_mode, norm, alpha=1.0, base_seed=0, alpha_range=None,
                 f_range=(0.0, 1.0), b_range=(0.0, 1.0)):
        super().__init__(obs_mode, norm, alpha, base_seed, alpha_range=alpha_range)
        self.f_range = f_range
        self.b_range = b_range

    def __call__(self, batch):
        idxs = [b[0] for b in batch]
        xs = torch.stack([b[1] for b in batch]).clone()   # (B, 2, L)
        ns = torch.stack([b[2] for b in batch])           # (B, L)
        cs = torch.stack([b[3] for b in batch])           # (B, L) continuum
        ys = torch.stack([b[4] for b in batch])           # (B, D)
        gen = self._gen(idxs)
        B = xs.shape[0]
        f = self.f_range[0] + (self.f_range[1] - self.f_range[0]) * torch.rand(B, 1, generator=gen)
        b = self.b_range[0] + (self.b_range[1] - self.b_range[0]) * torch.rand(B, 1, generator=gen)
        star = xs[:, 0, :] - xs[:, 1, :]                   # F_star (clouds don't touch it)
        p = xs[:, 1, :]
        p_cloud = (p + f * (cs - p)) * (1.0 + b * f)       # grey: mute bands + brighten
        xs[:, 1, :] = p_cloud
        xs[:, 0, :] = star + p_cloud                       # keep star_planet consistent
        a = self._alpha(B, gen)                            # exposure augmentation (after f,b)
        obs = inject_noise(xs, ns, self.obs_mode, alpha=a, generator=gen)
        return self.norm.encode(obs), ys
