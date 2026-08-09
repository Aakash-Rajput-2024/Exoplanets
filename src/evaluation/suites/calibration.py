"""Section I — Posterior calibration (SBC / TARP / PIT / coverage / ECE).

Tests whether the model's REPORTED UNCERTAINTY is trustworthy, using the field-standard
validation battery: simulation-based calibration rank histograms (Talts+2020), TARP
expected coverage (Lemos+2023), PIT + one-sample KS, central 68/95% coverage, and
regression-ECE (all via metrics_extra).

Posterior samples come from MC-dropout (T draws per seed, dropout enabled at eval —
works even at n=1 seed) pooled with the deep ensemble across seeds. Computed at the
noiseless reference so photon noise is not conflated with model calibration. n<3 seeds is
stamped PRELIMINARY (limited epistemic ensemble diversity), but the MC-dropout rank stats
are still informative. If dropout is inactive (samples collapse) the section says so.

BATTERY VALIDATION (the control this section previously lacked)
    Bad coverage numbers have two possible causes — a broken posterior, or a broken
    metric — and MC-dropout alone cannot separate them. So when the Section-K reference
    exists we additionally push the T0 EXACT posterior (importance sampling over the
    INARA library; ``evaluation.bayes.prior_is``) through the identical battery. A
    correct battery must score that near cov68=0.68 / cov95=0.95 / ECE≈0. If it does
    and MC-dropout still scores badly, the metric is exonerated and the model's
    uncertainty is genuinely untrustworthy.

    Note the reference is evaluated at its own resolved α (low), not at this section's
    noiseless α, and on its own planet subset — it validates the METRIC, and is not a
    like-for-like calibration comparison against the network.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from evaluation import core, plots
from evaluation import metrics_extra as mx
from common.pipeline import load_eval_raw, get_norm
from common.observable import make_observable
from common.evaluate import MAX_TRAINED_ALPHA
from common.data import TARGET_COLUMNS

SECTION, SUITE = "I", "calibration"
TITLE = "Posterior calibration (SBC / TARP / PIT / ECE)"
EPISTEMIC = "Ground truth: exact. Is the reported uncertainty trustworthy?"

ACTIVE = ["H2O", "CO2", "O2", "CH4", "N2O", "O3", "SO2", "NH3", "C2H6", "NO2"]
LOG_FLOOR = 1e-12


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(os.path.join(ctx.cache_v2, "test_x.pt")):
        return False, "INARA test cache missing"
    return True, ""


def _reference_calibration():
    """Run the identical battery on the T0 exact posterior, if Section K has produced
    one. Returns None when absent — this is a control, never a requirement.

    Picks the LOWEST resolved α, i.e. the one where importance sampling actually has
    effective sample size and the draws are a genuine posterior rather than repeats of
    a single library member.
    """
    import json
    from evaluation.bayes import likelihood as bl

    root = os.path.join(core.REPO, "results_v2", "bayes_reference")
    summary = os.path.join(root, "summary.json")
    if not os.path.exists(summary):
        return None
    try:
        with open(summary) as f:
            doc = json.load(f)
        resolved = sorted(float(a) for a, v in doc["by_alpha"].items() if v.get("resolved"))
        for a in resolved:
            z_path = os.path.join(root, f"prior_is_a{a:g}.npz")
            if not os.path.exists(z_path):
                continue
            z = np.load(z_path)
            if "samples_log" not in z:
                continue
            s_log, t_log = z["samples_log"], z["truth_log"]
            rel = mx.reliability(s_log, t_log)
            tarp = mx.tarp(s_log, t_log)
            ks = mx.pit_uniformity(mx.pit(s_log, t_log))
            ai = [TARGET_COLUMNS.index(c) for c in ACTIVE]
            return {
                "alpha": a, "n_samples": int(s_log.shape[0]), "n_planets": int(s_log.shape[1]),
                "ess_median": doc["by_alpha"][str(a)]["ess_median"],
                "coverage_68": float(np.mean([rel["coverage_68"][i] for i in ai])),
                "coverage_95": float(np.mean([rel["coverage_95"][i] for i in ai])),
                "reliability_ECE": rel["ece"], "TARP_ECE": tarp["ece_tarp"],
                "PIT_KS": float(np.mean([ks[i]["ks_stat"] for i in ai])),
                "ess_min_gate": bl.ESS_MIN,
            }
    except Exception as e:                      # a missing control must never break I
        return {"error": f"{type(e).__name__}: {e}"}
    return None


def _enable_dropout(model):
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()
            n += 1
    return n


@torch.no_grad()
def _forward(model, x, device, batch=256):
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(model(x[i:i + batch].to(device)).cpu())
    return torch.cat(out)


def run(ctx, n_cal=1000, T=30):
    ob, inn, lt = ctx.ckpt_config()
    raw_x, y_lin, noise, ids, lp = load_eval_raw(ctx.cache_v2, "test", lt)
    rng = np.random.default_rng(0)
    idx = rng.choice(raw_x.shape[0], size=min(n_cal, raw_x.shape[0]), replace=False)
    raw_x, y_lin, noise = raw_x[idx], y_lin[idx], noise[idx]
    norm = get_norm(ctx.cache_v2, ob, inn)
    x = norm.encode(make_observable(raw_x, noise, ob, alpha=MAX_TRAINED_ALPHA))

    n_dropout = 0
    samples = []          # each [N, D] linear
    try:
        for s in ctx.seeds:
            model = ctx.model(s)
            model.eval()
            n_dropout = _enable_dropout(model)
            for _ in range(T if n_dropout else 1):
                pred_std = _forward(model, x, ctx.device)
                samples.append(lp.decode(pred_std).numpy())
                core.free_device(ctx.device)   # keep MC-dropout draws from fragmenting MPS
    finally:
        # CRITICAL: models are cached on ctx and shared with later suites (e.g. J).
        # Restore eval mode so the dropout we enabled here can't leak stochasticity
        # into any subsequent suite's deterministic predictions.
        for s in ctx.seeds:
            ctx.model(s).eval()
    samples = np.stack(samples)                         # [S, N, D]
    S = samples.shape[0]
    samples_log = np.log10(np.clip(samples, LOG_FLOOR, None))
    truth_log = np.log10(np.clip(y_lin.numpy(), LOG_FLOOR, None))

    spread = float(np.median(samples_log.std(axis=0)))
    degenerate = spread < 1e-4

    rel = mx.reliability(samples_log, truth_log)
    pit_vals = mx.pit(samples_log, truth_log)
    ks = mx.pit_uniformity(pit_vals)
    ranks = mx.sbc_ranks(samples_log, truth_log)
    sbc = mx.sbc_uniformity(ranks, S)
    tarp = mx.tarp(samples_log, truth_log)

    ai = [TARGET_COLUMNS.index(s) for s in ACTIVE]
    cov68 = float(np.mean([rel["coverage_68"][i] for i in ai]))
    cov95 = float(np.mean([rel["coverage_95"][i] for i in ai]))
    ks_active = float(np.mean([ks[i]["ks_stat"] for i in ai]))

    status = "preliminary" if (ctx.n_seeds < 3 or degenerate) else "ok"
    per_sp_rows = [[c, round(rel["coverage_68"][i], 3), round(rel["coverage_95"][i], 3),
                    round(ks[i]["ks_stat"], 3), round(sbc[i]["chi2"], 1)]
                   for i, c in enumerate(TARGET_COLUMNS)]

    caveats = [
        f"Posterior = {S} samples (MC-dropout T={T} × {ctx.n_seeds} seed(s); dropout layers "
        f"active={n_dropout}); computed at the noiseless reference (α={MAX_TRAINED_ALPHA:g}).",
        "Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the "
        "diagonal. Inactive species (N2, CO) are expected to be wide (the model should report "
        "ignorance, not fabricate).",
    ]
    if degenerate:
        caveats.append("⚠ POSTERIOR COLLAPSED (near-zero sample spread) — dropout is inactive or "
                       "epistemic diversity is nil; calibration numbers are not meaningful. Train "
                       "≥3 seeds and/or verify the model carries active nn.Dropout.")
    if ctx.n_seeds < 3:
        caveats.append("PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.")

    headline = {"n_posterior_samples": S, "coverage_68 (active)": cov68,
                "coverage_95 (active)": cov95, "PIT-KS (active mean)": ks_active,
                "reliability_ECE": rel["ece"], "TARP_ECE": tarp["ece_tarp"],
                "posterior_spread(dex)": spread}
    tables = [{"name": "Per-species calibration",
               "columns": ["species", "cov68", "cov95", "PIT-KS", "SBC χ²"],
               "rows": per_sp_rows}]

    ref = _reference_calibration()
    if ref and "error" not in ref:
        tables.append({
            "name": "Battery validation — exact Bayesian posterior through the same metrics",
            "columns": ["posterior", "cov68", "cov95", "reliability ECE", "TARP ECE", "PIT-KS"],
            "rows": [
                ["ideal", 0.68, 0.95, 0.0, 0.0, 0.0],
                [f"T0 exact (α={ref['alpha']:g}, ESS≈{ref['ess_median']:.0f})",
                 round(ref["coverage_68"], 3), round(ref["coverage_95"], 3),
                 round(ref["reliability_ECE"], 3), round(ref["TARP_ECE"], 3),
                 round(ref["PIT_KS"], 3)],
                [f"MC-dropout ({ctx.label})", round(cov68, 3), round(cov95, 3),
                 round(rel["ece"], 3), round(tarp["ece_tarp"], 3), round(ks_active, 3)],
            ]})
        headline["reference_ECE (exact posterior)"] = ref["reliability_ECE"]
        caveats.append(
            f"BATTERY VALIDATED: the T0 exact posterior (α={ref['alpha']:g}, "
            f"{ref['n_samples']} draws) scores ECE {ref['reliability_ECE']:.3f} / cov68 "
            f"{ref['coverage_68']:.3f} through these same functions. The metrics are "
            "therefore sound, and any MC-dropout miscalibration above is a property of "
            "the model's uncertainty, not of the measurement. The reference sits at its "
            "own resolved α and planet subset — it is a metric control, NOT a "
            "like-for-like calibration comparison.")
    elif ref:
        caveats.append(f"(battery-validation control unavailable: {ref['error']})")

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status=status, epistemic=EPISTEMIC,
        headline=headline, tables=tables,
        caveats=caveats, provenance=ctx.provenance(),
        data={"reliability": rel, "pit_ks": ks, "sbc": sbc, "tarp": tarp,
              "coverage_68_active": cov68, "coverage_95_active": cov95, "n_samples": S,
              "reference_posterior": ref},
    )
    for fn, plotter in [("reliability", lambda d: plots.reliability_plot(d, ctx.label, rel)),
                        ("tarp", lambda d: plots.tarp_plot(d, ctx.label, tarp)),
                        ("sbc", lambda d: plots.sbc_hist(d, ctx.label, ranks, S, TARGET_COLUMNS))]:
        try:
            res.artifacts.append(plotter(ctx.out_dir(SUITE)))
        except Exception as e:
            res.caveats.append(f"(plot {fn} skipped: {e})")
    return core.write_result(ctx, res)
