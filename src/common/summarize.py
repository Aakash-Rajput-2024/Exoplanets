"""Collate all tracks' test-set results into one comparison (the C4 payoff).

Reads each track's ``eval_v2/summary.json`` (written by ``common.evaluate``) and
prints/saves a single table: overall test R² (log10) mean±std across seeds, per
architecture, all trained under the identical matched budget. Because the budget
is matched (registry.MATCHED), differences here ARE attributable to architecture
— which is exactly what C4 said the old comparison could not claim.

    python -m common.summarize
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.registry import TRACKS, track_config, MATCHED, REPO


def _summary_path(track):
    return os.path.join(os.path.dirname(track_config(track)["model_py"]), "eval_v2", "summary.json")


def _fmt_mean_std(r):
    if r.get("n_seeds", len(r.get("seeds", []))) > 1 and r.get("r2_log10_std") is not None:
        return f"{r['r2_log10_mean']:.4f} ± {r['r2_log10_std']:.4f}"
    return f"{r['r2_log10_mean']:.4f} (n=1)"


def collate(out_dir=None):
    out_dir = out_dir or os.path.join(REPO, "results_v2")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for track in TRACKS:
        p = _summary_path(track)
        if os.path.exists(p):
            with open(p) as f:
                rows.append(json.load(f))
    if not rows:
        raise SystemExit("No eval_v2/summary.json found — run common.evaluate first.")
    rows.sort(key=lambda r: r["r2_log10_mean"], reverse=True)

    # Report what ACTUALLY ran (stamped run_config), not the aspirational budget.
    def rc(r, k, default="?"):
        v = (r.get("run_config") or {}).get(k, default)
        return default if v is None else v

    def _fmtset(vals):  # sort-safe display of a mixed int/"?" set
        return sorted(set(vals), key=str)

    hashes = {rc(r, "config_hash") for r in rows if rc(r, "config_hash") != "?"}
    epochs_run = _fmtset(rc(r, "epochs_run") for r in rows)
    epochs_planned = _fmtset(rc(r, "epochs_planned") for r in rows)
    n_seeds = _fmtset(r.get("n_seeds", len(r.get("seeds", []))) for r in rows)
    any_undertrained = any(
        isinstance(rc(r, "epochs_run"), int) and isinstance(rc(r, "epochs_planned"), int)
        and rc(r, "epochs_run") < rc(r, "epochs_planned") for r in rows)
    any_single_seed = any((r.get("n_seeds", len(r.get("seeds", [])))) < 3 for r in rows)

    # Duplicate detection: identical outputs ⇒ not an independent architecture
    # result (e.g. causal ≡ transformerarch when the DSCM CFs were not applied).
    seen, dup_of = {}, {}
    for r in rows:
        key = (round(r["r2_log10_ensemble"], 5), round(r["rmse_dex"], 5), round(r["mae_dex"], 5))
        if key in seen:
            dup_of[r["track"]] = seen[key]
        else:
            seen[key] = r["track"]

    L = ["=" * 96,
         "ARCHITECTURE COMPARISON — FROZEN TEST SET, MATCHED BUDGET (ULTRAPLAN C4)",
         "=" * 96,
         f"Budget target (registry.MATCHED): loss={MATCHED['loss']} batch={MATCHED['batch_size']} "
         f"epochs={MATCHED['epochs']} lr={MATCHED['lr']} wd={MATCHED['weight_decay']} seeds={MATCHED['seeds']}",
         f"Observable={MATCHED['obs_mode']}  labels={MATCHED['label_transform']}  "
         f"input_norm={MATCHED['input_norm']}",
         f"AS-RUN: epochs_run={epochs_run} (planned {epochs_planned})  "
         f"seeds/track={n_seeds}  config_hash={'match' if len(hashes)<=1 else 'MISMATCH '+str(hashes)}"]
    if any_undertrained or any_single_seed:
        issues = []
        if any_undertrained:
            issues.append("undertrained (epochs_run < planned)")
        if any_single_seed:
            issues.append("single-seed (need ≥3 for CIs)")
        L.append(f"⚠ PRELIMINARY — some tracks are {', '.join(issues)}; "
                 "numbers below are NOT citable (rerun full budget × ≥3 seeds).")
    L += ["-" * 96,
          f"{'track':<16}| {'R2(log10)':<20}| {'ensemble':<9}| {'RMSE(dex)':<10}| {'MAE(dex)':<9}| desc",
          "-" * 96]
    for r in rows:
        note = f"  [≡ {dup_of[r['track']]}: identical outputs — not independent]" if r["track"] in dup_of else ""
        L.append(f"{r['track']:<16}| {_fmt_mean_std(r):<20}| {r['r2_log10_ensemble']:<9.4f}"
                 f"| {r['rmse_dex']:<10.4f}| {r['mae_dex']:<9.4f}| {r['desc']}{note}")
    L.append("=" * 96)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "architecture_comparison.txt"), "w") as f:
        f.write(txt)
    print(txt)

    names = [r["track"] for r in rows]
    means = [r["r2_log10_mean"] for r in rows]
    stds = [(r["r2_log10_std"] if r.get("r2_log10_std") is not None else 0.0) for r in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(names, means, yerr=stds, capsize=5, color="steelblue", alpha=0.85)
    plt.ylabel("Overall test R² (log10)")
    plt.title("Architecture comparison at matched budget (test set, mean±std over seeds)")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "architecture_comparison.png"), dpi=150)
    plt.close()
    print(f"=> wrote {out_dir}/architecture_comparison.{{txt,png}}")


if __name__ == "__main__":
    collate()
