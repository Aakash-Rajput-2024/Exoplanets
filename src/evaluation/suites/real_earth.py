"""Section F — REAL disk-integrated Earth (the only real-photon ground-truth test).

Uses the Robinson et al. 2011 VPL 3-D spectral Earth model at quadrature
(``data/real_earth/earth_quadrature_radiance_refl.dat``), validated against EPOXI and
earthshine, with Earth's real ~60% cloud cover baked in — the ideal real reflected-light
target because its composition is exactly known and it is a genuinely cloudy planet.

This is the v2-native successor to ``src/cloud_robust/realtest.py`` (v1: 1-channel,
scalar-standardised). Here the file's reflected-intensity column is the planet channel
and its solar-flux column is the star channel, so a proper 2-channel contrast+stellar-SNR
observable is built and scored through the identical v2 pipeline.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from evaluation import core, synth, known_targets, plots
from evaluation import metrics_extra as mx
from common.data import TARGET_COLUMNS

sys.path.insert(0, os.path.join(core.SRC, "evaluation/crossgen"))
from inara_grid import load_inara_grid   # noqa: E402

SECTION, SUITE = "F", "real_earth"
TITLE = "Real disk-integrated Earth (VPL Robinson 2011)"
EPISTEMIC = "Ground truth: LITERAL, on REAL photons. The strongest single real test."

EARTH_DAT = os.path.join(core.REPO, "data", "real_earth", "earth_quadrature_radiance_refl.dat")


def applicable(ctx):
    if not ctx.has_checkpoints:
        return False, "no checkpoints for this track/seed"
    if not os.path.exists(EARTH_DAT):
        return False, f"VPL Earth spectrum missing ({EARTH_DAT})"
    return True, ""


def _load_vpl():
    """Return (wl_um, intensity[planet], solar_flux[star]) from the Robinson file."""
    rows = []
    for ln in open(EARTH_DAT):
        ln = ln.strip()
        if not ln or ln.startswith(";"):
            continue
        try:
            rows.append([float(x) for x in ln.split()])
        except ValueError:
            pass
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]     # wl, intensity(W/m2/um/sr), solar_flux(W/m2/um)


def run(ctx):
    grid = load_inara_grid()
    ewl, eint, eflux = _load_vpl()
    planet = np.interp(grid, ewl, eint)
    star = np.interp(grid, ewl, eflux)

    raw_x, noise = synth.build_raw(planet[None], star[None], ctx.cache_v2, seed=0)
    ref = core.predict_raw(ctx, raw_x, noise, noiseless=True)
    a1 = core.predict_raw(ctx, raw_x, noise, alpha=1.0)
    pred_ref, pred_a1 = ref["ens"][0], a1["ens"][0]

    earth = next(t for t in known_targets.solar_system_targets() if t["name"] == "Earth")
    summ = mx.known_truth_summary(pred_ref, earth["true_vector"], TARGET_COLUMNS,
                                  earth["covered_species"])

    rows = [[c, f"{earth['true_vector'][i]:.2e}", f"{pred_ref[i]:.2e}", f"{pred_a1[i]:.2e}",
             "yes" if c in earth["covered_species"] else "-"]
            for i, c in enumerate(TARGET_COLUMNS)]

    artifacts = []
    try:
        artifacts.append(plots.truth_bars(ctx.out_dir(SUITE), "Earth-VPL", TARGET_COLUMNS,
                                          earth["true_vector"], pred_ref, earth["covered_species"]))
    except Exception:
        pass

    res = core.SuiteResult(
        SECTION, SUITE, TITLE, ctx.label, status="ok", epistemic=EPISTEMIC,
        headline={"dominant true": summ["dominant_true"], "dominant pred": summ["dominant_pred"],
                  "dominant correct": summ["dominant_correct"],
                  "mean dex-err (covered)": summ["mean_dex_error_covered"],
                  "ordering ρ": summ["spearman_ordering"]},
        tables=[{"name": "Earth composition: truth vs predicted",
                 "columns": ["species", "Earth VMR", "pred (noiseless)", "pred (α=1)", "covered"],
                 "rows": rows}],
        caveats=[
            "REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and "
            "median-matched to INARA before the INARA-fit norm — the observable/units are adapted, "
            "not identical to INARA.",
            "INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, "
            "O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases "
            "(N2, O2, H2O, CO2).",
            "Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into "
            "data/real_earth/ with the same loader.",
        ],
        artifacts=artifacts, provenance=ctx.provenance(),
        data={"summary": summ, "pred_noiseless": pred_ref.tolist(), "pred_alpha1": pred_a1.tolist()},
    )
    return core.write_result(ctx, res)
