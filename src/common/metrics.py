"""Correct retrieval metrics with reporting in log space (fixes C2).

Two fixes over the copied ``1 - ss_res/(ss_tot + 1e-8)``:

1. R² is undefined when ss_tot == 0; the additive epsilon silently biases the
   score toward 1 whenever the target variance is tiny (the trace-gas case in
   linear space). We return NaN there instead — an honest "undefined", not a
   fabricated 1.0.
2. Metrics are reported in **log10 mixing-ratio space** by default. There every
   species has O(1) variance, so per-species R²/RMSE are comparable and the
   epsilon issue is moot; RMSE/MAE read in *dex*, the natural unit of retrieval
   precision.

``report`` also returns 95% bootstrap CIs over samples (planets) so a 0.70-vs-
0.72 difference can be judged against its sampling noise (H6).
"""

from __future__ import annotations

import numpy as np

LOG_FLOOR = 1e-12


def r2_per_species(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def rmse_per_species(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))


def mae_per_species(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=0)


def to_space(y_linear, space):
    """Map linear mole fractions into the reporting space."""
    if space == "linear":
        return np.asarray(y_linear)
    if space == "log10":
        return np.log10(np.clip(np.asarray(y_linear), LOG_FLOOR, None))
    raise ValueError(f"unknown reporting space {space!r}")


def report(y_true_lin, y_pred_lin, space="log10", bootstrap=1000, seed=0):
    """Per-species R²/RMSE/MAE in `space`, with 95% bootstrap CIs on R².

    Inputs are LINEAR mole fractions (decode model output first). Returns a dict
    of length-D arrays: r2, rmse, mae, and (if bootstrap) r2_lo, r2_hi.
    """
    yt = to_space(y_true_lin, space)
    yp = to_space(y_pred_lin, space)
    out = {
        "space": space,
        "r2": r2_per_species(yt, yp),
        "rmse": rmse_per_species(yt, yp),
        "mae": mae_per_species(yt, yp),
    }
    if bootstrap:
        rng = np.random.default_rng(seed)
        n = yt.shape[0]
        draws = np.empty((bootstrap, yt.shape[1]))
        for b in range(bootstrap):
            idx = rng.integers(0, n, n)
            draws[b] = r2_per_species(yt[idx], yp[idx])
        with np.errstate(invalid="ignore"):
            out["r2_lo"] = np.nanpercentile(draws, 2.5, axis=0)
            out["r2_hi"] = np.nanpercentile(draws, 97.5, axis=0)
    return out
