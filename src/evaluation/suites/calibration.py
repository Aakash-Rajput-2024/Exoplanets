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

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status=status, epistemic=EPISTEMIC,
        headline={"n_posterior_samples": S, "coverage_68 (active)": cov68,
                  "coverage_95 (active)": cov95, "PIT-KS (active mean)": ks_active,
                  "reliability_ECE": rel["ece"], "TARP_ECE": tarp["ece_tarp"],
                  "posterior_spread(dex)": spread},
        tables=[{"name": "Per-species calibration",
                 "columns": ["species", "cov68", "cov95", "PIT-KS", "SBC χ²"],
                 "rows": per_sp_rows}],
        caveats=caveats, provenance=ctx.provenance(),
        data={"reliability": rel, "pit_ks": ks, "sbc": sbc, "tarp": tarp,
              "coverage_68_active": cov68, "coverage_95_active": cov95, "n_samples": S},
    )
    for fn, plotter in [("reliability", lambda d: plots.reliability_plot(d, ctx.label, rel)),
                        ("tarp", lambda d: plots.tarp_plot(d, ctx.label, tarp)),
                        ("sbc", lambda d: plots.sbc_hist(d, ctx.label, ranks, S, TARGET_COLUMNS))]:
        try:
            res.artifacts.append(plotter(ctx.out_dir(SUITE)))
        except Exception as e:
            res.caveats.append(f"(plot {fn} skipped: {e})")
    return core.write_result(ctx, res)
