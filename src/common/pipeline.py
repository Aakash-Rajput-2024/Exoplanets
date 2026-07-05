"""End-to-end observable→dataset composition (closes the C1/C2/H3 integration gap).

Until now `observable.py` (contrast + noise, C1), `inputs.py`
(PerLambdaAsinhNorm, H3) and `transforms.py` (CLR labels, C2) existed but nothing
tied them together — `data.load_v2` still did scalar per-channel standardization
of the raw *unobservable* planet flux. This module is the single seam every track
and evaluator now goes through:

    raw cache_v2 (star_planet, planet, noise, CLR-able labels)
        │  make_observable(mode)              # C1: physical observable
        │  PerLambdaAsinhNorm / ScalarNorm    # H3: per-λ asinh encoding
        │  LabelPipeline(clr)                  # C2: simplex-correct targets
        ▼
    train  = NoisyObservableDataset  (fresh seeded noise per epoch, C1)
    val/test = fixed seeded noisy observable at SNR-multiplier α

Design choices that matter for a fair comparison:
  * The input normalizer is fit on the **clean** train observable and cached, so
    every seed/architecture shares identical encoding statistics (no leakage,
    no per-run drift).
  * The val/test noise realizations are seeded with **constants independent of
    the model seed**, so all seeds/architectures are scored on the exact same
    noisy spectra — differences are the model's, not the noise draw's.
  * Only the *training* noise stream depends on the model seed (per-seed
    augmentation diversity, which is what the multi-seed study wants).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from common.data import load_raw, label_pipeline, DERIVED_CACHE_VERSION
from common.observable import make_observable
from common.inputs import (
    RawObservableDataset,
    RawCloudObservableDataset,
    NoiseCollate,
    CloudNoiseCollate,
    build_eval_observable,
    fit_norm,
    norm_from_state,
)
from common import cloud_families

# Eval-time noise seeds — fixed so every model is scored on identical spectra.
VAL_NOISE_SEED = 101
TEST_NOISE_SEED = 202


def norm_cache_path(cache_dir, input_norm, obs_mode):
    return os.path.join(cache_dir, f"norm_{input_norm}_{obs_mode}_{DERIVED_CACHE_VERSION}.pt")


def get_norm(cache_dir, obs_mode="contrast_snr", input_norm="per_lambda_asinh"):
    """Train-fitted input normalizer for (obs_mode, input_norm), cached to disk.

    Fit on the CLEAN train observable (no noise) so the encoding statistics are a
    fixed property of the data, reused for val/test and across every seed.
    """
    path = norm_cache_path(cache_dir, input_norm, obs_mode)
    if os.path.exists(path):
        return norm_from_state(torch.load(path))
    tr_x, _, tr_noise, _ = load_raw(cache_dir, "train", feature_mode="both")
    clean = make_observable(tr_x, tr_noise, obs_mode)
    norm = fit_norm(input_norm, clean)
    torch.save(norm.state(), path)
    return norm


def get_train_planet_continuum(cache_dir):
    """Band-free continuum of the TRAIN planet channel, cached to disk.

    Needed by the grey-cloud training augmentation (C5). Estimated once on the
    clean planet flux; clouds do not change the continuum, so it is reused across
    epochs/seeds.
    """
    path = os.path.join(cache_dir, f"continuum_train_planet_{DERIVED_CACHE_VERSION}.pt")
    if os.path.exists(path):
        return torch.load(path)
    x, _, _, _ = load_raw(cache_dir, "train", feature_mode="both")
    planet = x[:, 1, :].numpy()
    cont = torch.from_numpy(cloud_families.estimate_continuum(planet)).float()
    torch.save(cont, path)
    return cont


@dataclass
class PipelineData:
    train_ds: object
    val_ds: object
    test_ds: object
    label_pipe: object
    norm: object
    in_channels: int
    seq_len: int
    obs_mode: str
    train_collate: object = None   # vectorized batch collate for the train loader


def _eval_dataset(raw_x, noise, y_enc, obs_mode, norm, alpha, seed, alpha_range=None):
    x = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=alpha, seed=seed,
                              alpha_range=alpha_range)
    return TensorDataset(x, y_enc)


def build_datasets(
    cache_dir,
    obs_mode="contrast_snr",
    label_transform="clr",
    input_norm="per_lambda_asinh",
    alpha=1.0,
    alpha_train_range=None,
    train_seed=0,
    noise_train=True,
    cloud_train=None,
    splits=("train", "val", "test"),
):
    """Build the requested split datasets under the observable/label/norm config.

    ``train_seed`` seeds ONLY the training noise augmentation (per-seed diversity);
    val/test use fixed constant seeds so all models see identical noisy spectra.
    ``cloud_train="grey"`` additionally applies the grey cloud family to the train
    planet channel each epoch (C5 training family). ``alpha_train_range`` (lo, hi)
    turns on per-sample exposure-time augmentation for TRAIN and VAL (one model
    spans the whole SNR sweep); TEST stays at fixed ``alpha`` (each sweep point is
    one exposure). Returns a :class:`PipelineData`; unrequested splits are ``None``.
    """
    norm = get_norm(cache_dir, obs_mode, input_norm)
    lp = label_pipeline(cache_dir, label_transform)

    train_ds = val_ds = test_ds = None
    train_collate = None
    in_channels = seq_len = None

    if "train" in splits:
        x, y, noise, _ = load_raw(cache_dir, "train", feature_mode="both")
        y_enc = lp.encode(y)
        if cloud_train == "grey":
            cont = get_train_planet_continuum(cache_dir)
            train_ds = RawCloudObservableDataset(x, noise, cont, y_enc)
            train_collate = CloudNoiseCollate(obs_mode, norm, alpha=alpha, base_seed=train_seed,
                                              alpha_range=alpha_train_range)
        elif noise_train:
            train_ds = RawObservableDataset(x, noise, y_enc)
            train_collate = NoiseCollate(obs_mode, norm, alpha=alpha, base_seed=train_seed,
                                         alpha_range=alpha_train_range)
        else:
            enc = norm.encode(make_observable(x, noise, obs_mode))
            train_ds = TensorDataset(enc, y_enc)
        if train_collate is not None:
            train_collate.set_epoch(0)
            probe_x, _ = train_collate([train_ds[0], train_ds[1]])
            in_channels, seq_len = probe_x.shape[1], probe_x.shape[2]
        else:
            probe = train_ds[0][0]
            in_channels, seq_len = probe.shape[0], probe.shape[1]

    if "val" in splits:
        x, y, noise, _ = load_raw(cache_dir, "val", feature_mode="both")
        # VAL is augmented like TRAIN so best-val selection targets average-over-
        # exposure performance, not the near-zero α=1 photon floor.
        val_ds = _eval_dataset(x, noise, lp.encode(y), obs_mode, norm, alpha, VAL_NOISE_SEED,
                               alpha_range=alpha_train_range)
        if in_channels is None:
            in_channels, seq_len = val_ds[0][0].shape

    if "test" in splits:
        x, y, noise, _ = load_raw(cache_dir, "test", feature_mode="both")
        test_ds = _eval_dataset(x, noise, lp.encode(y), obs_mode, norm, alpha, TEST_NOISE_SEED)
        if in_channels is None:
            in_channels, seq_len = test_ds[0][0].shape

    return PipelineData(
        train_ds, val_ds, test_ds, lp, norm, in_channels, seq_len, obs_mode, train_collate
    )


def load_eval_raw(cache_dir, split="test", label_transform="clr"):
    """Raw handles for the SNR sweep: (raw_x[N,2,L], y_linear[N,12], noise[N,L]).

    The evaluator re-forms the observable at each α itself, so it needs the raw
    star_planet/planet/noise, plus the label pipeline to decode predictions.
    """
    x, y, noise, ids = load_raw(cache_dir, split, feature_mode="both")
    lp = label_pipeline(cache_dir, label_transform)
    return x, y, noise, ids, lp
