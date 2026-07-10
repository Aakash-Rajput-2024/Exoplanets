"""Unified evaluation driver.

Runs every APPLICABLE suite (A…J) for a trained checkpoint through the shared v2 core and
writes one consolidated report + a cross-track leaderboard:

    src/evaluation/results/<track>/report.md      — human report, one section each
    src/evaluation/results/<track>/summary.json   — machine roll-up (all headlines)
    src/evaluation/results/leaderboard.json       — accumulated across tracks/seeds

A suite whose inputs are absent is recorded as *skipped* (offline-safe) — the run never
fails as a whole. Heavy compute (training seeds, generating large engine caches) is done
by the user; this only evaluates existing checkpoints/caches.

Usage:
    python -m evaluation.run_eval <track> [--seeds 0 1 2] [--suffix _grey]
                                          [--suites all | A C E F I | in_distribution ...]
                                          [--alphas 0.3 1 3 10 30 100 300]
                                          [--n-cal 1000] [--T 30]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import traceback

from evaluation import core, registry
from common.evaluate import DEFAULT_ALPHAS


def _run_suite(mod, ctx, kwargs):
    ok, reason = mod.applicable(ctx)
    if not ok:
        return core.SuiteResult.skipped(mod.SECTION, mod.SUITE, mod.TITLE, ctx.label, reason)
    try:
        # pass only kwargs the suite's run() accepts
        import inspect
        params = inspect.signature(mod.run).parameters
        kw = {k: v for k, v in kwargs.items() if k in params}
        return mod.run(ctx, **kw)
    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-3:]
        return core.SuiteResult(mod.SECTION, mod.SUITE, mod.TITLE, ctx.label,
                                status="error", caveats=["; ".join(tb)])


def _write_report(ctx, results):
    prov = ctx.provenance()
    d = os.path.join(core.RESULTS_ROOT, ctx.label)
    os.makedirs(d, exist_ok=True)
    L = [f"# Evaluation report — `{ctx.label}`", "",
         f"*generated {datetime.datetime.now().isoformat(timespec='seconds')}*  ·  "
         f"seeds {ctx.seeds}  ·  device `{ctx.device}`  ·  "
         f"ran {prov.get('epochs_run')}/{prov.get('epochs_planned')} epochs  ·  "
         f"git `{str(prov.get('git_commit'))[:12]}`", "",
         "> Reflected-light / direct-imaging retrieval (0.2–2.0 µm, LUVOIR-like). "
         "Each section states its epistemic status. The only LITERAL-ground-truth real tests are "
         "Sections E (solar-system) and F (real Earth); transiting-planet data (G, and 'far' rows "
         "of H) is a wrong-observable OOD probe, not an accuracy measurement.", "",
         "## Contents", ""]
    for r in results:
        icon = core._STATUS_ICON.get(r.status, "")
        L.append(f"- **{r.section}** {r.title} — {icon} {r.status}")
    L.append("")
    for r in results:
        L.append(core.render_markdown(r))
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    summary = {"track": ctx.label, "seeds": ctx.seeds, "n_seeds": ctx.n_seeds,
               "provenance": prov,
               "sections": {r.section: {"suite": r.suite, "status": r.status,
                                        "headline": r.headline} for r in results}}
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=core._json_default)
    return os.path.join(d, "report.md"), summary


def _update_leaderboard(summary):
    os.makedirs(core.RESULTS_ROOT, exist_ok=True)
    path = os.path.join(core.RESULTS_ROOT, "leaderboard.json")
    board = {}
    if os.path.exists(path):
        with open(path) as f:
            board = json.load(f)
    board[summary["track"]] = {
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_seeds": summary["n_seeds"],
        "sections": {s: v["headline"] for s, v in summary["sections"].items()
                     if v["status"] in ("ok", "preliminary")},
    }
    with open(path, "w") as f:
        json.dump(board, f, indent=2, default=core._json_default)
    return path


def main():
    ap = argparse.ArgumentParser(description="Unified evaluation pipeline (sections A–J).")
    ap.add_argument("track")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--suites", nargs="*", default=["all"],
                    help="section letters (A C E), slugs (in_distribution), or 'all'")
    ap.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    ap.add_argument("--cache", default=core.CACHE_V2)
    ap.add_argument("--n-cal", type=int, default=1000, help="calibration: planets sampled")
    ap.add_argument("--T", type=int, default=30, help="calibration: MC-dropout draws/seed")
    a = ap.parse_args()

    ctx = core.EvalContext(track=a.track, seeds=a.seeds, suffix=a.suffix,
                           alphas=tuple(a.alphas), cache_v2=a.cache)
    if not ctx.has_checkpoints:
        raise SystemExit(f"No checkpoints for {a.track}{a.suffix} at seeds {a.seeds} "
                         f"(looked for model_best_seed*{a.suffix}.pth).")
    mods = registry.resolve(a.suites)
    kwargs = {"n_cal": a.n_cal, "T": a.T}

    print(f"== evaluating {ctx.label}  seeds={ctx.seeds}  suites={[m.SECTION for m in mods]} ==")
    results = []
    for mod in mods:
        print(f"  [{mod.SECTION}] {mod.SUITE} ...", flush=True)
        r = _run_suite(mod, ctx, kwargs)
        print(f"      -> {r.status}" + (f": {r.caveats[0]}" if r.status in ("skipped", "error")
                                        and r.caveats else ""))
        results.append(r)

    report_path, summary = _write_report(ctx, results)
    board_path = _update_leaderboard(summary)
    n_ok = sum(r.status in ("ok", "preliminary") for r in results)
    print(f"\n== {n_ok}/{len(results)} sections produced results ==")
    print(f"report:      {report_path}")
    print(f"leaderboard: {board_path}")


if __name__ == "__main__":
    main()
