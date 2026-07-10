"""Section G — Transiting-planet OOD probe (NOT an accuracy test).

Real transiting-planet spectra (JWST/HST transmission or emission of hot Jupiters) are a
DIFFERENT observable, wavelength range and planet class than this reflected-light
direct-imaging model. Feeding them is scientifically meaningless as retrieval — this
section exists only to (a) demonstrate that, and (b) provide a distance-to-training-
distribution anomaly score and prediction-stability read on genuinely off-distribution
inputs, with a loud disclaimer.

Runs fully offline on a band-template transmission DEMO (a hot-Jupiter-like spectrum from
reflected_engine.transmission_shape). Real spectra dropped into ``data/real_spectra/``
(2–3 col: wavelength_µm, depth/flux, [err]) are picked up automatically.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from evaluation import core, synth, known_targets
from evaluation import metrics_extra as mx
from evaluation.engines import reflected_engine as eng

sys.path.insert(0, os.path.join(core.SRC, "crossgen_eval"))
from inara_grid import load_inara_grid   # noqa: E402

SECTION, SUITE = "G", "transiting_ood"
TITLE = "Transiting-planet OOD probe"
EPISTEMIC = "NO ground truth. Wrong observable — anomaly/stability only, never accuracy."

REAL_DIR = os.path.join(core.REPO, "data", "real_spectra")
_STABILITY_SEEDS = [202, 303, 404, 505]


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    return True, ""     # always runs (offline demo)


def _stellar_planck(T, wl):
    return eng.stellar_continuum_shape({"star_temperature": T}, wl)


def _probe(ctx, planet_ch, star_ch, mu_ref, sd_ref):
    raw_x, noise = synth.build_raw(planet_ch[None], star_ch[None], ctx.cache_v2, seed=0)
    enc = core.encode_raw(ctx, raw_x, noise, noiseless=True)
    anom = float(mx.anomaly_score(enc, mu_ref, sd_ref)[0])
    preds = [core.predict_raw(ctx, raw_x, noise, alpha=1.0, seed=s)["ens"][0]
             for s in _STABILITY_SEEDS]
    stab = float(np.mean(np.std(np.stack(preds), axis=0)))       # mean per-species pred std
    return anom, stab


def _load_real(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln[0] in "#;":
            continue
        parts = ln.replace(",", " ").split()
        try:
            rows.append([float(x) for x in parts[:2]])
        except ValueError:
            pass
    a = np.array(rows)
    return (a[:, 0], a[:, 1]) if a.size else (None, None)


def run(ctx):
    wl = load_inara_grid()
    # INARA reference distribution for the anomaly score (encoded contrast channel)
    enc_inara, _, _, _, _ = core.encode_cache(ctx, ctx.cache_v2, noiseless=True)
    mu_ref, sd_ref = enc_inara[:, 0, :].mean(0), enc_inara[:, 0, :].std(0)
    inara_anom = mx.anomaly_score(enc_inara, mu_ref, sd_ref)
    inara_ref = {"median": float(np.median(inara_anom)), "p95": float(np.percentile(inara_anom, 95))}

    rows, data = [], {}
    # (1) offline demo — WASP-39b-like transmission spectrum (wrong observable)
    wasp = next(t for t in known_targets.published_targets() if t["name"] == "WASP-39b")
    demo_planet = eng.transmission_shape(wasp["planet"], wl)
    demo_star = _stellar_planck(wasp["planet"]["star_temperature"], wl)
    anom, stab = _probe(ctx, demo_planet, demo_star, mu_ref, sd_ref)
    rows.append(["DEMO: WASP-39b-like transmission", round(anom, 2),
                 f"{anom / inara_ref['median']:.1f}×", round(stab, 4)])
    data["demo_transmission"] = {"anomaly": anom, "stability_std": stab}

    # (2) any real spectra provided by the user
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "*.txt")) +
                        glob.glob(os.path.join(REAL_DIR, "*.csv")) +
                        glob.glob(os.path.join(REAL_DIR, "*.dat")))
    for f in real_files:
        rwl, rflux = _load_real(f)
        if rwl is None:
            continue
        planet = np.interp(wl, rwl, rflux)
        star = _stellar_planck(5500.0, wl)
        anom, stab = _probe(ctx, planet, star, mu_ref, sd_ref)
        name = os.path.basename(f)
        rows.append([f"REAL: {name}", round(anom, 2), f"{anom / inara_ref['median']:.1f}×",
                     round(stab, 4)])
        data[name] = {"anomaly": anom, "stability_std": stab}

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={"INARA anomaly median": inara_ref["median"], "INARA anomaly p95": inara_ref["p95"],
                  "real spectra found": len(real_files)},
        tables=[{"name": "OOD probe — anomaly vs INARA training distribution",
                 "columns": ["input", "anomaly (RMS z)", "× INARA median", "pred stability (std)"],
                 "rows": rows}],
        caveats=[
            "⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light "
            "direct-imaging model — any predicted composition is meaningless. The point is to "
            "SHOW the inputs are far off-distribution (large anomaly) and/or unstable.",
            "A high anomaly × INARA-median is the expected, desired result: the model correctly "
            "sees a transiting spectrum as out-of-distribution.",
            "To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into "
            "data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).",
        ],
        provenance=ctx.provenance(),
        data={"inara_reference": inara_ref, "probes": data},
    )
    return core.write_result(ctx, res)
