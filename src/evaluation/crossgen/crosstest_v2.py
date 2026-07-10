"""Cross-generator evaluation of the v2 (leak-free) models on petitRADTRANS.

Reuses the EXACT v2 eval machinery (common.pipeline / common.inputs /
common.observable / common.evaluate internals) so the trained weights are scored
identically to the INARA test set — same 2-channel contrast+SNR observable, same
INARA-train-fit per-λ asinh norm, same CLR label decode, same SNR sweep — but on
a pRT-generated test split (data/cache_v2_prt from build_cache_v2_prt.py).

It writes to src/evaluation/crossgen/results_v2_prt/<track><suffix>/ and NEVER touches
each track's INARA eval_v2/ directory.

The headline is the COVERED-species mean R² (H2O, CO2, O2, CH4, O3 — the molecules
pRT actually imprints on the spectrum). The other 7 INARA targets carry real
labels but no pRT spectral signal, so their R² is uninformative by construction.

Because α=1 is the photon-starved LUVOIR floor (planet below noise ⇒ R²≈0 BY
PHYSICS, even on INARA), the informative domain-shift number is the NOISELESS
CONTRAST REF (zero contrast noise, SNR channel held at MAX_TRAINED_ALPHA — see
common.evaluate; NOT the true α→∞ limit, which would saturate
observable.SNR_CLAMP and be OOD) and the SNR sweep — compare these to the
INARA-test references.

VARIANTS excludes "2channel1dcnn": it is architecturally identical to
"original1dcnn" (bit-identical predictions), so listing both double-counted one
result as two independent rows. main() also runtime-checks predictions across
whatever variants ARE run and flags (does not silently drop) any further
duplicates via the "duplicate_of" field in summary_all.json.

Run in venvNLP (torch):
  python src/evaluation/crossgen/crosstest_v2.py                 # all tracks + transformer grey
  python src/evaluation/crossgen/crosstest_v2.py --only causal
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
sys.path.insert(0, os.path.join(REPO, "src"))

from common.registry import track_config
from common.runtime import get_device
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED
from common.inputs import build_eval_observable
from common.observable import snr_summary, make_observable
from common import metrics
from common.evaluate import (_load_model, _predict, _ckpt_path, _ckpt_provenance,
                             DEFAULT_ALPHAS, MAX_TRAINED_ALPHA, TARGET_COLUMNS)

COVERED = ["H2O", "CO2", "O2", "CH4", "O3"]   # pRT line species ∩ INARA targets
COV_IDX = [TARGET_COLUMNS.index(s) for s in COVERED]
DEFAULT_CACHE = os.path.join(REPO, "data", "cache_v2_prt")
RESULTS_ROOT = os.path.join(REPO, "src", "evaluation/crossgen", "results_v2_prt")

# (track, checkpoint suffix) — user choices: v2 leak-free for all; transformer all variants.
# NOTE: "2channel1dcnn" was REMOVED (was previously listed alongside
# "original1dcnn") -- the two tracks share the identical model architecture and
# produce bit-identical predictions, so listing both double-counted a single
# result as two independent architecture rows. See also the runtime duplicate
# guard in main() below, which flags (rather than silently drops) any other
# variant pair whose predictions turn out identical.
VARIANTS = [
    ("original1dcnn", ""),
    ("causal", ""),
    ("transformerarch", ""),
    ("transformerarch", "_grey"),
]


def _covmean(r2):
    return float(np.nanmean(np.asarray(r2)[COV_IDX]))


def evaluate_variant(track, suffix, cache_dir, alphas, seed=0):
    label = f"{track}{suffix}"
    if not os.path.exists(_ckpt_path(track, seed, suffix)):
        print(f"[skip] {label}: no checkpoint model_best_seed{seed}{suffix}.pth")
        return None
    cfg = track_config(track)
    device = get_device()
    model, ck_cfg = _load_model(track, seed, device, suffix)
    obs_mode = ck_cfg.get("obs_mode", cfg["obs_mode"])
    input_norm = ck_cfg.get("input_norm", cfg["input_norm"])

    raw_x, y_lin, noise, _ids, lp = load_eval_raw(cache_dir, "test", cfg["label_transform"])
    y_true = y_lin.numpy()
    norm = get_norm(cache_dir, obs_mode, input_norm)

    def r2_at(x_obs):
        pred = lp.decode(_predict(model, x_obs, device)).numpy()
        return metrics.report(y_true, pred, space="log10", bootstrap=0), pred

    # headline at α=1 (photon-starved floor) with bootstrap CIs
    x1 = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=1.0, seed=TEST_NOISE_SEED)
    pred1 = lp.decode(_predict(model, x1, device)).numpy()
    rep = metrics.report(y_true, pred1, space="log10", bootstrap=1000)

    # SNR sweep (covered + all-12 R² per α)
    sweep = []
    for a in alphas:
        xa = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=a, seed=TEST_NOISE_SEED)
        rep_a, _ = r2_at(xa)
        row = dict(alpha=float(a), **snr_summary(raw_x[:, 0], raw_x[:, 1], noise, a))
        row["r2_covered"] = _covmean(rep_a["r2"])
        row["r2_all12"] = float(np.nanmean(rep_a["r2"]))
        sweep.append(row)

    # Noiseless reference (domain-shift headline). Uses MAX_TRAINED_ALPHA, NOT
    # CEILING_ALPHA (=1e4): CEILING_ALPHA saturates every bin's SNR channel
    # against observable.SNR_CLAMP=(1e-3,1e3), which is an OOD input the model
    # never trained on (same clamp-saturation bug as common.evaluate; fixed
    # there by introducing MAX_TRAINED_ALPHA, reused here for consistency).
    x_ceil = norm.encode(make_observable(raw_x, noise, obs_mode, alpha=MAX_TRAINED_ALPHA))
    rep_ceil, pred_ceil = r2_at(x_ceil)
    ceiling = {"r2_covered": _covmean(rep_ceil["r2"]),
               "r2_all12": float(np.nanmean(rep_ceil["r2"]))}

    prov = _ckpt_provenance(track, seed, device, suffix)
    out_dir = os.path.join(RESULTS_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "track": track, "suffix": suffix, "engine": "prt", "seed": seed,
        "covered_species": COVERED,
        "r2_covered_alpha1": _covmean(rep["r2"]),
        "r2_all12_alpha1": float(np.nanmean(rep["r2"])),
        "r2_covered_ceiling": ceiling["r2_covered"],
        "r2_all12_ceiling": ceiling["r2_all12"],
        "per_species_r2_alpha1": {c: float(rep["r2"][i]) for i, c in enumerate(TARGET_COLUMNS)},
        "snr_sweep": sweep, "ceiling": ceiling,
        "obs_mode": obs_mode, "input_norm": input_norm, "run_config": prov,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_report(out_dir, label, cfg, rep, sweep, ceiling, obs_mode, input_norm, prov)
    _plot_sweep(out_dir, label, sweep, ceiling)
    _plot_scatter(out_dir, label, y_true, pred_ceil, rep_ceil)
    print(f"=> {label}: covered R²(log10) ceiling={ceiling['r2_covered']:.4f}  "
          f"α=1={_covmean(rep['r2']):.4f}   (wrote {out_dir})")
    # pred_ceil (per-species predictions at the noiseless reference) is returned
    # alongside the summary so main() can detect variants whose predictions are
    # bit-identical to an already-evaluated one (e.g. original1dcnn vs.
    # 2channel1dcnn were the same model under two names) and flag them instead
    # of silently double-counting a single result as two independent rows.
    return summary, pred_ceil


def _write_report(out_dir, label, cfg, rep, sweep, ceiling, obs_mode, input_norm, prov):
    L = ["=" * 80,
         "CROSS-GENERATOR EVALUATION (v2 leak-free model) — petitRADTRANS test split",
         "=" * 80,
         f"Track:        {label}  ({cfg['desc']})",
         f"Train gen:    NASA PSG / INARA     Eval gen: petitRADTRANS (reflected+scattering)",
         f"Observable:   {obs_mode}   Input norm: {input_norm} (INARA-train fit)   Labels: {cfg['label_transform']}",
         f"Ran:          {prov.get('epochs_run')}/{prov.get('epochs_planned')} epochs  "
         f"best@{prov.get('best_epoch')}  git {str(prov.get('git_commit'))[:12]}",
         f"Covered species (pRT-imprinted): {', '.join(COVERED)}",
         "-" * 80,
         "HEADLINE — covered-species mean R²(log10):",
         f"  noiseless contrast ref (α={MAX_TRAINED_ALPHA:g}, in-distribution, NOT α→∞): "
         f"{ceiling['r2_covered']:.4f}   [all-12: {ceiling['r2_all12']:.4f}]",
         f"  α=1 (LUVOIR floor):      {_covmean(rep['r2']):.4f}   "
         f"[all-12: {float(np.nanmean(rep['r2'])):.4f}]",
         "  (α=1 R²≈0 is expected BY PHYSICS — planet below noise; read the ref/sweep.)",
         "-" * 80,
         "PER-SPECIES @ α=1 (95% bootstrap CI on R²):",
         f"{'Species':<8}| {'cov':<4}| {'R2':<9}| {'95% CI':<18}| {'RMSE(dex)':<10}| {'MAE(dex)':<9}"]
    L.append("-" * 80)
    for i, col in enumerate(TARGET_COLUMNS):
        ci = f"[{rep['r2_lo'][i]:.2f}, {rep['r2_hi'][i]:.2f}]"
        flag = "yes" if col in COVERED else "-"
        L.append(f"{col:<8}| {flag:<4}| {rep['r2'][i]:<9.4f}| {ci:<18}| {rep['rmse'][i]:<10.4f}| {rep['mae'][i]:<9.4f}")
    L += ["-" * 80,
          "SNR SWEEP — covered-species R² vs exposure (α=√(t/t_nom)):",
          f"{'alpha':<7}| {'SNR_planet(band)':<17}| {'R2 covered':<11}| {'R2 all-12':<10}"]
    L.append("-" * 80)
    for r in sweep:
        L.append(f"{r['alpha']:<7}| {r['snr_planet_band']:<17.3f}| {r['r2_covered']:<11.4f}| {r['r2_all12']:<10.4f}")
    L.append("=" * 80)
    with open(os.path.join(out_dir, "crossgen_details_prt.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


def _plot_sweep(out_dir, label, sweep, ceiling):
    xs = [r["snr_planet_band"] for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [r["r2_covered"] for r in sweep], "o-", color="navy", label="covered species")
    ax.plot(xs, [r["r2_all12"] for r in sweep], "s--", color="gray", alpha=0.6, label="all 12")
    ax.axhline(ceiling["r2_covered"], color="darkorange", ls="--", alpha=0.8,
               label=f"covered noiseless ref ({ceiling['r2_covered']:.3f})")
    ax.axvline(5.0, color="crimson", ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Band-integrated planet detection SNR (median)")
    ax.set_ylabel("R² (log10)")
    ax.set_title(f"{label}: cross-gen (pRT) retrieval vs SNR")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "r2_vs_snr_prt.png"), dpi=150)
    plt.close(fig)


def _plot_scatter(out_dir, label, y_true, y_pred, rep):
    yt = np.log10(np.clip(y_true, 1e-12, None))
    yp = np.log10(np.clip(y_pred, 1e-12, None))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for i, ax in enumerate(axes.flatten()):
        cov = TARGET_COLUMNS[i] in COVERED
        ax.scatter(yt[:, i], yp[:, i], alpha=0.3, s=8,
                   color="royalblue" if cov else "lightgray")
        lo = min(yt[:, i].min(), yp[:, i].min()); hi = max(yt[:, i].max(), yp[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--")
        tag = "" if cov else " (uncovered)"
        ax.set_title(f"{TARGET_COLUMNS[i]}{tag}  R²={rep['r2'][i]:.3f}")
        ax.set_xlabel("true log10 X"); ax.set_ylabel("pred log10 X")
    fig.suptitle(f"{label}: pRT cross-gen, noiseless contrast ref (α={MAX_TRAINED_ALPHA:g})", y=1.001)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_true_prt_ceiling.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    ap.add_argument("--only", default=None, help="run a single track (e.g. causal)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    variants = [(t, s) for (t, s) in VARIANTS if a.only in (None, t)]
    results = []
    evaluated_preds = []  # [(label, pred_ceil ndarray), ...] for the dedup guard below
    for track, suffix in variants:
        out = evaluate_variant(track, suffix, a.cache, a.alphas, seed=a.seed)
        if not out:
            continue
        r, pred_ceil = out
        label = f"{track}{suffix}"
        # Guard against double-counting: if this variant's per-species
        # predictions are bit-identical (within float tolerance) to one already
        # evaluated in this run, it is not an independent architecture result
        # (this is exactly how original1dcnn/2channel1dcnn duplicated a single
        # row before 2channel1dcnn was removed from VARIANTS above). Flag it in
        # the summary and printed table rather than silently reporting it as if
        # it were new information.
        dup_of = None
        for prev_label, prev_pred in evaluated_preds:
            if np.allclose(pred_ceil, prev_pred, rtol=1e-6, atol=1e-9):
                dup_of = prev_label
                break
        r["duplicate_of"] = dup_of
        if dup_of:
            print(f"[flag] {label}: predictions identical to {dup_of} — "
                  f"NOT an independent architecture result, excluded from ranking")
        evaluated_preds.append((label, pred_ceil))
        results.append(r)
    if results:
        print("\n" + "=" * 72)
        print("CROSS-GEN (pRT) SUMMARY — covered-species mean R²(log10)")
        print(f"{'variant':<24}| {'ceiling (α→∞)':<15}| {'α=1 (floor)':<12}")
        print("-" * 72)
        for r in results:
            v = f"{r['track']}{r['suffix']}"
            note = f"  [≡ {r['duplicate_of']}: not independent]" if r.get("duplicate_of") else ""
            print(f"{v:<24}| {r['r2_covered_ceiling']:<15.4f}| {r['r2_covered_alpha1']:<12.4f}{note}")
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        with open(os.path.join(RESULTS_ROOT, "summary_all.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {os.path.join(RESULTS_ROOT, 'summary_all.json')}")


if __name__ == "__main__":
    main()
