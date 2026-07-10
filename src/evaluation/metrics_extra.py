"""Evaluation metrics beyond ``common.metrics`` — the field-standard tools the
retrieval-ML literature uses that the base R²/RMSE reporter doesn't cover.

Everything here is pure-numpy (no torch/model coupling) so it is unit-testable and
reusable across suites. Three families:

  1. KNOWN-TRUTH, small-n  (solar-system, real Earth, published targets): dex error,
     "did it get the dominant gas", abundance-ordering Spearman. Reported instead of
     R² when there are too few targets for a variance-based score.

  2. OOD HONESTY (VALIDATION_PLAN B6): per-λ standardized offset δ(λ) and variance
     ratio v(λ) of a generator vs INARA after the INARA-fit norm; and per-species
     R²_raw | R²_debiased | r²_pearson so a negative cross-gen R² is decomposed into
     bias vs genuine loss of ordering.

  3. POSTERIOR CALIBRATION (Talts+2020 SBC; Lemos+2023 TARP; PIT/coverage/ECE): given
     posterior SAMPLES in log10-VMR space [S, N, D], test whether the reported
     uncertainty is trustworthy. Sampler-agnostic — the calibration suite feeds it a
     deep ensemble and/or MC-dropout draws.

Reference: Talts et al. 2020 (arXiv:1804.06788, SBC); Lemos et al. 2023
(arXiv:2302.03026, TARP); Vasist et al. 2023 A&A 672 A147 (NPE + coverage for
exoplanet retrieval).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _sps
except Exception:                       # scipy optional; degrade to numpy-only paths
    _sps = None

LOG_FLOOR = 1e-12


def _log10(x):
    return np.log10(np.clip(np.asarray(x, dtype=float), LOG_FLOOR, None))


# =========================================================================== #
# 1. Known-truth, small-n
# =========================================================================== #
def dex_error(pred_lin, true_lin):
    """|log10(pred) − log10(true)| per species (the natural retrieval-precision unit)."""
    return np.abs(_log10(pred_lin) - _log10(true_lin))


def dominant_gas(vmr_lin, target_columns):
    return target_columns[int(np.argmax(np.asarray(vmr_lin)))]


def dominant_correct(pred_lin, true_lin):
    return int(np.argmax(np.asarray(pred_lin))) == int(np.argmax(np.asarray(true_lin)))


def abundance_spearman(pred_lin, true_lin):
    """Spearman ρ between predicted and true abundance ORDERING across species —
    'did it rank which gases are abundant', robust to absolute-scale bias."""
    pt, tt = _log10(pred_lin), _log10(true_lin)
    if _sps is not None:
        rho = _sps.spearmanr(pt, tt).correlation
        return float(rho) if rho == rho else 0.0
    pr, tr = _rankdata(pt), _rankdata(tt)             # numpy fallback
    return float(np.corrcoef(pr, tr)[0, 1])


def _rankdata(a):
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a))
    return ranks


def known_truth_summary(pred_lin, true_lin, target_columns, covered=None):
    """Per-target truth table + headline for a SINGLE known-composition target.

    ``covered`` = species the forward model imprints (dex error is only meaningful
    where the spectrum carries the band). Giants/uncovered species become an
    honesty check (should not be confidently high).
    """
    pred_lin = np.asarray(pred_lin, dtype=float)
    true_lin = np.asarray(true_lin, dtype=float)
    de = dex_error(pred_lin, true_lin)
    cov_mask = np.array([c in (covered or target_columns) for c in target_columns])
    return {
        "dominant_true": dominant_gas(true_lin, target_columns),
        "dominant_pred": dominant_gas(pred_lin, target_columns),
        "dominant_correct": dominant_correct(pred_lin, true_lin),
        "spearman_ordering": abundance_spearman(pred_lin, true_lin),
        "mean_dex_error_covered": float(np.mean(de[cov_mask])) if cov_mask.any() else None,
        "mean_dex_error_all": float(np.mean(de)),
        "per_species": {
            c: {"true": float(true_lin[i]), "pred": float(pred_lin[i]),
                "dex_error": float(de[i]), "covered": bool(cov_mask[i])}
            for i, c in enumerate(target_columns)
        },
    }


# =========================================================================== #
# 2. OOD honesty (B6)
# =========================================================================== #
def standardized_shift(enc_gen, enc_inara, channel=0):
    """Per-λ standardized offset δ(λ)=(μ_gen−μ_INARA)/σ_INARA and variance ratio
    v(λ)=σ²_gen/σ²_INARA on an ENCODED observable channel (after the INARA-fit norm).

    δ≈0, v≈1 ⇒ the generator sits inside INARA's input distribution at that λ; large
    |δ| or v far from 1 localises the domain shift in wavelength.
    """
    g = np.asarray(enc_gen)[:, channel, :]
    a = np.asarray(enc_inara)[:, channel, :]
    mu_a, sd_a = a.mean(0), a.std(0) + 1e-12
    delta = (g.mean(0) - mu_a) / sd_a
    vratio = (g.var(0) + 1e-12) / (a.var(0) + 1e-12)
    return {
        "delta_lambda": delta, "v_lambda": vratio,
        "median_abs_delta": float(np.median(np.abs(delta))),
        "median_v_ratio": float(np.median(vratio)),
        "frac_lambda_in_family": float(np.mean((np.abs(delta) <= 0.5) &
                                               (vratio >= 0.5) & (vratio <= 2.0))),
    }


def debiased_r2(true_lin, pred_lin, target_columns, covered=None):
    """R²_raw | R²_debiased | r²_pearson per species (log10 space).

    R²_debiased removes only the constant per-species bias b=median(pred−true) — so a
    negative raw R² that becomes ≥0 after debiasing is *biased extrapolation* (fixable
    by domain-robust normalisation), whereas one that stays negative is genuine loss of
    information. r²_pearson (squared Pearson) is the best-affine ordering score.
    """
    yt, yp = _log10(true_lin), _log10(pred_lin)
    out = {}
    for i, c in enumerate(target_columns):
        t, p = yt[:, i], yp[:, i]
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2_raw = 1 - np.sum((t - p) ** 2) / ss_tot if ss_tot > 0 else np.nan
        p_db = p - np.median(p - t)
        r2_db = 1 - np.sum((t - p_db) ** 2) / ss_tot if ss_tot > 0 else np.nan
        if t.std() > 0 and p.std() > 0:
            r_pear = float(np.corrcoef(t, p)[0, 1]) ** 2
        else:
            r_pear = np.nan
        out[c] = {"r2_raw": float(r2_raw), "r2_debiased": float(r2_db),
                  "r2_pearson": r_pear, "bias_dex": float(np.median(p - t))}
    cov = covered or target_columns
    m = lambda k: float(np.nanmean([out[c][k] for c in cov]))          # noqa: E731
    return {"per_species": out, "covered": list(cov),
            "mean_r2_raw": m("r2_raw"), "mean_r2_debiased": m("r2_debiased"),
            "mean_r2_pearson": m("r2_pearson")}


# =========================================================================== #
# 3. Posterior calibration — samples[S,N,D] in log10-VMR space, truth[N,D]
# =========================================================================== #
def pit(samples_log, truth_log):
    """Probability-integral-transform value per (planet, species): F_post(truth)."""
    s = np.asarray(samples_log)
    t = np.asarray(truth_log)[None, :, :]
    return (s <= t).mean(axis=0)                    # [N, D]


def pit_uniformity(pit_vals):
    """One-sample KS of pooled PIT vs Uniform(0,1) per species (flat ⇒ calibrated)."""
    out = {}
    for d in range(pit_vals.shape[1]):
        x = pit_vals[:, d]
        if _sps is not None:
            ks = _sps.kstest(x, "uniform")
            out[d] = {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}
        else:
            xs = np.sort(x)
            cdf = np.arange(1, len(xs) + 1) / len(xs)
            out[d] = {"ks_stat": float(np.max(np.abs(xs - cdf))), "ks_p": None}
    return out


def central_coverage(samples_log, truth_log, level):
    """Empirical coverage of the central ``level`` credible interval per species."""
    s = np.asarray(samples_log)
    t = np.asarray(truth_log)
    lo = np.quantile(s, (1 - level) / 2, axis=0)
    hi = np.quantile(s, (1 + level) / 2, axis=0)
    return ((t >= lo) & (t <= hi)).mean(axis=0)      # [D]


def reliability(samples_log, truth_log, grid=None):
    """Reliability curve (nominal vs empirical central coverage) + regression-ECE."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else np.asarray(grid)
    emp = np.stack([central_coverage(samples_log, truth_log, g) for g in grid])  # [G, D]
    emp_mean = emp.mean(axis=1)
    ece = float(np.mean(np.abs(emp_mean - grid)))
    return {"nominal": grid.tolist(), "empirical_mean": emp_mean.tolist(),
            "empirical_per_species": emp.tolist(), "ece": ece,
            "coverage_68": central_coverage(samples_log, truth_log, 0.68).tolist(),
            "coverage_95": central_coverage(samples_log, truth_log, 0.95).tolist()}


def sbc_ranks(samples_log, truth_log):
    """SBC rank of the truth among posterior samples per (planet, species), 0..S.
    Uniform over {0..S} ⇒ calibrated; ∩ (peaked centre) ⇒ over-dispersed; ∪ (peaked
    edges) ⇒ over-confident."""
    s = np.asarray(samples_log)
    t = np.asarray(truth_log)[None, :, :]
    return (s < t).sum(axis=0)                       # [N, D] ints in 0..S


def sbc_uniformity(ranks, n_samples, n_bins=20):
    """χ² of the SBC rank histogram vs uniform per species (large ⇒ mis-calibrated)."""
    out = {}
    for d in range(ranks.shape[1]):
        counts, _ = np.histogram(ranks[:, d], bins=n_bins, range=(0, n_samples + 1))
        exp = counts.sum() / n_bins
        chi2 = float(np.sum((counts - exp) ** 2 / exp)) if exp > 0 else np.nan
        if _sps is not None and exp > 0:
            p = float(1 - _sps.chi2.cdf(chi2, n_bins - 1))
        else:
            p = None
        out[d] = {"chi2": chi2, "p": p, "hist": counts.tolist()}
    return out


def tarp(samples_log, truth_log, n_alpha=21, n_refs=1, seed=0):
    """TARP expected-coverage test (Lemos+2023), joint over all D species.

    DROP variant: standardise dims by pooled posterior std; for each example draw a
    reference point, measure the fraction of posterior mass closer to the reference
    than the truth; the empirical CDF of that fraction vs the nominal credibility
    level is the coverage curve. Diagonal ⇒ calibrated; above ⇒ conservative; below ⇒
    over-confident. ECE_TARP = mean|coverage − nominal|.
    """
    s = np.asarray(samples_log, dtype=float)         # [S, N, D]
    t = np.asarray(truth_log, dtype=float)           # [N, D]
    S, N, D = s.shape
    scale = s.reshape(-1, D).std(axis=0) + 1e-12
    s = s / scale
    t = t / scale
    rng = np.random.default_rng(seed)
    alpha = np.linspace(0, 1, n_alpha)
    covs = []
    pooled = s.reshape(-1, D)
    for _ in range(max(1, n_refs)):
        ref = pooled[rng.integers(0, pooled.shape[0], size=N)]        # [N, D]
        d_truth = np.linalg.norm(t - ref, axis=1)                     # [N]
        d_samp = np.linalg.norm(s - ref[None], axis=2)               # [S, N]
        f = (d_samp < d_truth[None]).mean(axis=0)                    # [N] posterior mass inside
        covs.append([(f <= a).mean() for a in alpha])
    cov = np.mean(covs, axis=0)
    return {"alpha": alpha.tolist(), "coverage": cov.tolist(),
            "ece_tarp": float(np.mean(np.abs(cov - alpha)))}


def anomaly_score(enc_obs, mu_ref, sd_ref, channel=0):
    """Mahalanobis-like distance-to-training-distribution per spectrum (diagonal
    covariance from the INARA test observable). Used by the OOD probe to say 'how
    far off-distribution is this real input' without claiming any accuracy."""
    x = np.asarray(enc_obs)[:, channel, :]
    z = (x - mu_ref) / (sd_ref + 1e-12)
    return np.sqrt(np.mean(z ** 2, axis=1))          # [N] RMS z per spectrum
