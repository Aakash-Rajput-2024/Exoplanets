"""Classical (non-neural) baselines for the retrieval task — a reference point
so the neural R² numbers elsewhere in the paper are interpretable.

Uses the EXACT same data/metric machinery as the neural evaluator
(``common.evaluate``), so the numbers are apples-to-apples:
  * ``common.pipeline.load_eval_raw`` for the raw (star_planet, planet, noise,
    labels) tensors, feature_mode "both" (2-channel).
  * ``common.inputs.build_eval_observable`` to build the SAME contrast_snr
    observable at alpha=1, reusing the TRAIN-fit input normalizer cached by
    ``common.pipeline.get_norm`` (never refit here — no leakage).
  * ``common.metrics.report`` to score in log10 mixing-ratio space, exactly
    like ``common.evaluate.evaluate_track``, so R2_log10 numbers are directly
    comparable to the per-track eval_v2/summary.json files.

BASELINES
  (a) PriorMean   — predicts the TRAIN mean CLR-space abundance for every test
                     planet, ignoring the spectrum entirely. This is the
                     "using no information at all" reference; its R2_log10
                     is ~0 by construction (it IS the SS_tot baseline used by
                     R2), so it calibrates what "R2 = 0" means in this task.
  (b) Ridge       — multi-output linear regression on a dimensionality-reduced
                     observable.
  (c) RandomForest — multi-output (native, via scikit-learn's
                     RandomForestRegressor multi-output support) on the same
                     reduced features.

DIMENSIONALITY REDUCTION
    The raw contrast_snr observable is (N, 2, 4378) = 8756 features; with
    ~88k train planets that's a ~88k x 8756 dense float matrix (~3 GB) which is
    wasteful to feed to Ridge/RF as-is and pointless (both channels are smooth
    in wavelength, so neighboring bins are redundant). We AVERAGE-POOL each
    channel from 4378 -> N_BINS contiguous wavelength bins (default 64),
    giving a 2*N_BINS = 128-dim feature vector per planet. This is preferred
    here over a fitted PCA because it needs no extra fitted artifact beyond
    what's already cached (norm, label pipeline): it's a fixed, deterministic,
    physically-interpretable reduction (coarse spectral bins) that requires
    no additional train-only statistics to manage or risk leaking.

TARGETS
    CLR space (the label pipeline's default / MATCHED budget's
    label_transform), so Ridge/RF regress onto the same standardized,
    simplex-correct 12-dim target the neural nets are trained on. Predictions
    are decoded back through the SAME LabelPipeline.decode (CLR -> softmax)
    before scoring, exactly like evaluate.py.

Usage:
    python src/common/baselines.py [--n-subsample N] [--n-bins 64]
                                    [--cache data/cache_v2] [--skip-rf]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
REPO = os.path.abspath(os.path.join(SRC, os.pardir))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common.registry import CACHE_V2, MATCHED
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED
from common.inputs import build_eval_observable
from common import metrics

TARGET_COLUMNS = ["H2O", "CO2", "O2", "N2", "CH4", "N2O",
                  "CO", "O3", "SO2", "NH3", "C2H6", "NO2"]

OBS_MODE = MATCHED["obs_mode"]              # "contrast_snr"
INPUT_NORM = MATCHED["input_norm"]          # "per_lambda_asinh"
LABEL_TRANSFORM = MATCHED["label_transform"]  # "clr"
ALPHA = MATCHED["alpha"]                    # 1.0 — same headline alpha as evaluate.py

OUT_DIR = os.path.join(REPO, "results_v2", "baselines")


def pool_features(x_obs: torch.Tensor, n_bins: int = 64) -> np.ndarray:
    """Average-pool an encoded observable (N, C, L) -> (N, C*n_bins).

    Fixed, deterministic reduction (contiguous wavelength bins, adaptive-avg
    pooled so it works for any L without requiring L % n_bins == 0). No
    fitting involved, so it cannot leak train stats into test.
    """
    t = x_obs if torch.is_tensor(x_obs) else torch.as_tensor(x_obs)
    pooled = torch.nn.functional.adaptive_avg_pool1d(t, n_bins)  # (N, C, n_bins)
    return pooled.reshape(pooled.shape[0], -1).numpy()


def _subsample(n_total: int, n_subsample: int | None, seed: int = 0) -> np.ndarray:
    if n_subsample is None or n_subsample >= n_total:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_subsample, replace=False)


def build_features_and_labels(cache_dir, split, norm, lp, n_bins, n_subsample=None,
                              alpha=ALPHA, seed=TEST_NOISE_SEED):
    """Load a split, build the contrast_snr observable at alpha (fixed seed,
    matching evaluate.py's headline alpha=1 test/train materialization), pool
    it, and return (X_pooled, y_linear, y_clr_encoded)."""
    raw_x, y_lin, noise, _ids, _lp = load_eval_raw(cache_dir, split, LABEL_TRANSFORM)
    idx = _subsample(raw_x.shape[0], n_subsample, seed=0)
    raw_x, y_lin, noise = raw_x[idx], y_lin[idx], noise[idx]
    x_obs = build_eval_observable(raw_x, noise, OBS_MODE, norm, alpha=alpha, seed=seed)
    X = pool_features(x_obs, n_bins=n_bins)
    y_lin_np = y_lin.numpy()
    y_enc = lp.encode(y_lin).numpy()  # CLR-standardized target, same space nets train on
    return X, y_lin_np, y_enc


def fit_prior_mean(y_enc_train: np.ndarray) -> np.ndarray:
    """The TRAIN mean in (standardized) CLR space — constant prediction."""
    return y_enc_train.mean(axis=0, keepdims=True)


def run_baselines(cache_dir=CACHE_V2, n_bins=64, n_subsample=None, skip_rf=False,
                  rf_n_estimators=100, rf_max_depth=16, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    lp = load_eval_raw(cache_dir, "train", LABEL_TRANSFORM)[4]  # cached LabelPipeline
    norm = get_norm(cache_dir, OBS_MODE, INPUT_NORM)  # TRAIN-fit normalizer, cached (no refit)

    print(f"[baselines] loading train split (feature_mode=both, obs_mode={OBS_MODE})...")
    Xtr, ytr_lin, ytr_enc = build_features_and_labels(
        cache_dir, "train", norm, lp, n_bins, n_subsample=n_subsample, seed=0)
    print(f"[baselines] loading test split...")
    Xte, yte_lin, yte_enc = build_features_and_labels(
        cache_dir, "test", norm, lp, n_bins, n_subsample=n_subsample, seed=TEST_NOISE_SEED)
    print(f"[baselines] train X={Xtr.shape} y={ytr_enc.shape}  test X={Xte.shape} y={yte_enc.shape}"
         f"  ({time.time()-t0:.1f}s)")

    results = {}

    # (a) PriorMean: constant = train-mean CLR vector for every test planet.
    prior = fit_prior_mean(ytr_enc)
    pred_enc = np.repeat(prior, Xte.shape[0], axis=0)
    pred_lin = lp.decode(torch.as_tensor(pred_enc, dtype=torch.float32)).numpy()
    results["PriorMean"] = metrics.report(yte_lin, pred_lin, space="log10", bootstrap=0)

    # (b) Ridge on pooled features, multi-output natively supported.
    t1 = time.time()
    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(Xtr, ytr_enc)
    pred_enc = ridge.predict(Xte)
    pred_lin = lp.decode(torch.as_tensor(pred_enc, dtype=torch.float32)).numpy()
    results["Ridge"] = metrics.report(yte_lin, pred_lin, space="log10", bootstrap=0)
    print(f"[baselines] Ridge fit+predict: {time.time()-t1:.1f}s")

    # (c) RandomForest on pooled features. RandomForestRegressor natively supports
    # 2D y (one shared forest predicting all 12 outputs at each leaf) -- this is
    # NOT wrapped in MultiOutputRegressor, which would fit 12 independent forests
    # (~12x the trees) for no accuracy benefit here.
    if not skip_rf:
        t1 = time.time()
        rf = RandomForestRegressor(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            n_jobs=-1, random_state=seed,
        )
        rf.fit(Xtr, ytr_enc)
        pred_enc = rf.predict(Xte)
        pred_lin = lp.decode(torch.as_tensor(pred_enc, dtype=torch.float32)).numpy()
        results["RandomForest"] = metrics.report(yte_lin, pred_lin, space="log10", bootstrap=0)
        print(f"[baselines] RandomForest fit+predict: {time.time()-t1:.1f}s")

    _write_report(results, n_bins, Xtr.shape[0], Xte.shape[0], n_subsample)
    return results


def _write_report(results, n_bins, n_train, n_test, n_subsample):
    summary = {
        "obs_mode": OBS_MODE, "input_norm": INPUT_NORM, "label_transform": LABEL_TRANSFORM,
        "alpha": ALPHA, "n_bins_per_channel": n_bins, "n_train": n_train, "n_test": n_test,
        "n_subsample": n_subsample,
        "baselines": {},
    }
    lines = []
    lines.append("=" * 80)
    lines.append("CLASSICAL BASELINES — reference point for neural R2_log10 (ULTRAPLAN v2)")
    lines.append("=" * 80)
    lines.append(f"Observable: {OBS_MODE}  alpha={ALPHA}  input_norm={INPUT_NORM}  "
                 f"labels={LABEL_TRANSFORM}")
    lines.append(f"Features:   average-pooled to {n_bins} bins/channel "
                 f"({n_bins * 2}-dim input to Ridge/RF)")
    lines.append(f"n_train={n_train}  n_test={n_test}"
                 + (f"  (SUBSAMPLED from full cache, n_subsample={n_subsample})"
                    if n_subsample else "  (FULL split)"))
    lines.append(f"Reporting space:  log10 mixing ratio (dex); matches common.evaluate exactly")
    lines.append("-" * 80)
    for name, rep in results.items():
        overall_r2 = float(np.nanmean(rep["r2"]))
        overall_rmse = float(np.nanmean(rep["rmse"]))
        overall_mae = float(np.nanmean(rep["mae"]))
        summary["baselines"][name] = {
            "overall_r2_log10": overall_r2,
            "overall_rmse_dex": overall_rmse,
            "overall_mae_dex": overall_mae,
            "per_species_r2": {c: float(rep["r2"][i]) for i, c in enumerate(TARGET_COLUMNS)},
            "per_species_rmse_dex": {c: float(rep["rmse"][i]) for i, c in enumerate(TARGET_COLUMNS)},
        }
        lines.append(f"{name}:  overall R2(log10) = {overall_r2:.4f}   "
                     f"RMSE(dex) = {overall_rmse:.4f}   MAE(dex) = {overall_mae:.4f}")
        lines.append(f"{'Species':<9}| {'R2(log10)':<10}| {'RMSE(dex)':<10}| {'MAE(dex)':<9}")
        for i, col in enumerate(TARGET_COLUMNS):
            lines.append(f"{col:<9}| {rep['r2'][i]:<10.4f}| {rep['rmse'][i]:<10.4f}| {rep['mae'][i]:<9.4f}")
        lines.append("-" * 80)
    lines.append("=" * 80)
    txt = "\n".join(lines) + "\n"
    print(txt)

    with open(os.path.join(OUT_DIR, "details.txt"), "w") as f:
        f.write(txt)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[baselines] wrote {OUT_DIR}/summary.json and details.txt")


def main():
    ap = argparse.ArgumentParser(description="Classical (non-neural) baselines for the retrieval task.")
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--n-bins", type=int, default=64,
                    help="average-pool bins per channel (feature dim = 2*n_bins)")
    ap.add_argument("--n-subsample", type=int, default=None,
                    help="subsample train/test to this many planets each, for a quick check")
    ap.add_argument("--skip-rf", action="store_true", help="skip RandomForest (Ridge+PriorMean only)")
    ap.add_argument("--rf-n-estimators", type=int, default=100,
                    help="default 100 (not sklearn's 100... explicit here for clarity); "
                         "kept modest so the RF baseline stays a cheap classical fit, "
                         "not a compute-heavy search")
    ap.add_argument("--rf-max-depth", type=int, default=16,
                    help="cap tree depth so RF stays a cheap baseline fit on the full "
                         "~88k-planet train split (unconstrained depth on 128 pooled "
                         "features over 88k rows is needlessly slow for a reference "
                         "baseline and barely changes R2; see baselines.py docstring)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_baselines(cache_dir=a.cache, n_bins=a.n_bins, n_subsample=a.n_subsample,
                 skip_rf=a.skip_rf, rf_n_estimators=a.rf_n_estimators,
                 rf_max_depth=a.rf_max_depth, seed=a.seed)


if __name__ == "__main__":
    main()
