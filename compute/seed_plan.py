#!/usr/bin/env python3
"""
Seed-completion planner for the multi-seed study (H6).

Answers three questions in one pass:

  1. WHICH (track, seed) runs are still missing, and why (never run / truncated /
     trained under a different objective / trained on different hardware).
  2. HOW LONG the remaining work will take, calibrated from the per-epoch ``sec``
     field already recorded in every ``logs_v2/*.json``.
  3. WHAT to put in the paper -- emits a provenance table (device, epochs, wall
     clock, objective flags) so the training budget is reportable rather than
     reconstructed from memory.

Usage
-----
    PYTHONPATH=src python compute/seed_plan.py                 # plan + estimate
    PYTHONPATH=src python compute/seed_plan.py --strict        # also redo cross-device seeds
    PYTHONPATH=src python compute/seed_plan.py --device mac    # estimate for local MPS
    PYTHONPATH=src python compute/seed_plan.py --emit-manifest # write remaining_runs.tsv

The manifest is what ``compute/submit_remaining.sh`` consumes; this script never
launches training itself.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stat
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from common.registry import TRACKS, MATCHED, REPO  # noqa: E402

TARGET_SEEDS = list(MATCHED["seeds"])
TARGET_EPOCHS = MATCHED["epochs"]

# Objective flag per track. The suffix train_runner derives from it decides the
# checkpoint filename, so getting this wrong silently trains the wrong model.
OBJECTIVE = {
    "original1dcnn":   [],
    "optimized1dcnn":  [],
    "transformerarch": [],
    "causal":          ["--cf"],
    "causal_cfi":      ["--cf-invariance"],
    "causal_xl":       ["--cf-invariance"],
}
SUFFIX = {
    "causal": "_cf",
    "causal_cfi": "_cfi",
    "causal_xl": "_cfi",
}

# Track directory on disk (not always <src>/models/<track>).
TRACK_DIR = {
    "causal": "models/causal/cnn_trnas",
    "causal_cfi": "models/causal/cnn_trnas",
    "causal_xl": "models/causal/cnn_trnas",
}

# Fallback per-epoch seconds for (track, device) pairs never yet measured.
# Derived by scaling a measured anchor on the same device; see --explain.
FALLBACK_SCALE = {
    # track -> cost relative to optimized1dcnn on the same device
    "original1dcnn":   1.2,   # 36M params, but the FC head is cheap on GPU
    "optimized1dcnn":  1.0,
    "transformerarch": 3.2,   # attention over the downsampled sequence
    "causal":          6.0,   # transformer + a second forward pass on x^do(e)
    "causal_cfi":      6.0,
    "causal_xl":       13.0,
}


def track_dir(track: str) -> str:
    return os.path.join(REPO, "src", TRACK_DIR.get(track, f"models/{track}"))


def device_of(model_py: str) -> str:
    """Infer where a run happened from the absolute path baked into its config."""
    if "/scratch/" in model_py or "/home/" in model_py:
        return "hpc"
    if "/Users/" in model_py:
        return "mac"
    return "unknown"


def load_runs():
    """Every training run on disk, keyed by (track, seed, cloud)."""
    runs = defaultdict(list)
    for track in TRACKS:
        logs = os.path.join(track_dir(track), "logs_v2")
        if not os.path.isdir(logs):
            continue
        for name in sorted(os.listdir(logs)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(logs, name)) as fh:
                d = json.load(fh)
            if d.get("track") != track:
                continue  # sibling track sharing the directory
            cfg = d.get("config", {})
            hist = d.get("history", [])
            secs = [e["sec"] for e in hist if e.get("sec")]
            runs[(track, d["seed"], cfg.get("cloud_train"))].append(dict(
                file=name,
                epochs=len(hist),
                device=device_of(cfg.get("model_py", "")),
                cf_train=cfg.get("cf_train"),
                cf_invariance=cfg.get("cf_invariance"),
                sec_median=stat.median(secs) if secs else None,
                wall_hours=sum(secs) / 3600.0 if secs else None,
                best_val=d.get("best_val_loss"),
                final_r2=hist[-1].get("val_r2_log10") if hist else None,
            ))
    return runs


def objective_matches(track: str, run: dict) -> bool:
    """Was this run trained under the objective the track is supposed to use?"""
    want_cf = "--cf" in OBJECTIVE[track]
    want_inv = "--cf-invariance" in OBJECTIVE[track]
    # A missing key means the flag was absent when the run was launched.
    return bool(run.get("cf_train")) == want_cf and bool(run.get("cf_invariance")) == want_inv


def checkpoint_for(track: str, seed: int, cloud=None) -> str | None:
    """Locate the best checkpoint under either the legacy or current naming."""
    ck = os.path.join(track_dir(track), "checkpoints_v2")
    suf = SUFFIX.get(track, "") + (f"_{cloud}" if cloud else "")
    cands = [f"model_best_{track}_seed{seed}{suf}.pth",   # current naming
             f"model_best_seed{seed}{suf}.pth"]           # legacy naming
    if suf:
        # Pre-suffix runs wrote a bare name. For causal_cfi/causal_xl that bare
        # name collides between the two tracks, so only trust it where the track
        # is the sole occupant of its checkpoint directory.
        if track not in ("causal_cfi", "causal_xl"):
            cands.append(f"model_best_seed{seed}.pth")
    for cand in cands:
        p = os.path.join(ck, cand)
        if os.path.exists(p):
            return p
    return None


def resume_ckpt(track: str, seed: int, cloud=None) -> str | None:
    """The exact path train_runner --resume looks for; anything else is ignored.

    Older runs wrote `last_seed{S}.pth` without the track prefix, so a truncated
    run under the legacy name CANNOT be resumed and must restart from epoch 0.
    """
    suf = SUFFIX.get(track, "") + (f"_{cloud}" if cloud else "")
    p = os.path.join(track_dir(track), "checkpoints_v2", f"last_{track}_seed{seed}{suf}.pth")
    return p if os.path.exists(p) else None


def ckpt_progress(track: str, seed: int, cloud=None):
    """Epochs completed according to the resume checkpoint, or None.

    A run killed by the OOM reaper (SIGKILL) never reaches the code that writes
    logs_v2/*.json, so the log understates progress badly -- while the resume
    snapshot is fully up to date. Reading it back is the only way to price the
    remaining work correctly after a hard kill.
    """
    p = resume_ckpt(track, seed, cloud)
    if p is None:
        return None
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return len(ck.get("history") or []) or ck.get("next_epoch")
    except Exception:
        return None  # numpy/torch version skew; fall back to the logs


def classify(track, seed, runs, strict, ref_device):
    """Return (status, reason, epochs_done) for one (track, seed)."""
    candidates = runs.get((track, seed, None), [])
    valid = [r for r in candidates if objective_matches(track, r)]

    if not candidates:
        # No log yet does NOT mean no progress: a run that is still in flight, or
        # one killed before it could write logs_v2/*.json, has only its resume
        # snapshot to show for potentially many hours of compute.
        ck_done = ckpt_progress(track, seed)
        if ck_done:
            if ck_done >= TARGET_EPOCHS - 2:
                return "DONE", f"{ck_done} epochs (ckpt, log pending)", ck_done
            return "IN-PROGRESS", f"{ck_done}/{TARGET_EPOCHS} epochs (ckpt)", ck_done
        return "MISSING", "never trained", 0
    if not valid:
        # A stale log from a wrong-objective run does not rule out a correct one
        # having been started since: the checkpoint carries the objective in its
        # filename suffix, so if one exists it is by construction the right model.
        ck_done = ckpt_progress(track, seed)
        if ck_done:
            if ck_done >= TARGET_EPOCHS - 2:
                return "DONE", f"{ck_done} epochs (ckpt, log pending)", ck_done
            return "IN-PROGRESS", f"{ck_done}/{TARGET_EPOCHS} epochs (ckpt)", ck_done
        got = candidates[0]
        have = "cf" if got.get("cf_train") else ("cfi" if got.get("cf_invariance") else "plain")
        want = "cf" if "--cf" in OBJECTIVE[track] else (
            "cfi" if "--cf-invariance" in OBJECTIVE[track] else "plain")
        return "WRONG-OBJ", f"trained as '{have}', track needs '{want}'", 0

    best = max(valid, key=lambda r: r["epochs"])
    # Trust whichever source is further along: the log, or the resume snapshot.
    done = best["epochs"]
    ck_done = ckpt_progress(track, seed)
    from_ckpt = ck_done is not None and ck_done > done
    if from_ckpt:
        done = ck_done
    if done < TARGET_EPOCHS - 2:  # early-stopping tolerance
        if resume_ckpt(track, seed):
            src = "ckpt" if from_ckpt else "log"
            return "TRUNCATED", f"{done}/{TARGET_EPOCHS} epochs ({src}), resumable", done
        return ("RESTART", f"{best['epochs']}/{TARGET_EPOCHS} epochs, no resumable "
                           f"last_{track}_seed{seed}{SUFFIX.get(track, '')}.pth", 0)
    if checkpoint_for(track, seed) is None:
        return "NO-CKPT", f"{best['epochs']} epochs logged, checkpoint absent", 0
    if strict and best["device"] != ref_device:
        return "CROSS-DEV", f"trained on {best['device']}, study reference is {ref_device}", 0
    # Report `done`, not best["epochs"]: when the checkpoint is ahead of the log
    # (run finished but was killed before writing logs_v2/*.json) the log count is
    # stale and would understate a completed run.
    src = " (ckpt, log stale)" if from_ckpt else f" on {best['device']}"
    return "DONE", f"{done} epochs{src}", done


def calibrate(runs, device):
    """Measured median s/epoch per track on `device`, plus a scaled fallback."""
    measured = {}
    for (track, _seed, cloud), rs in runs.items():
        if cloud is not None:
            continue
        vals = [r["sec_median"] for r in rs
                if r["device"] == device and r["sec_median"] and objective_matches(track, r)]
        if vals:
            measured.setdefault(track, []).extend(vals)
    # Median across seeds is robust to one contended node (causal_xl seed 2).
    measured = {t: stat.median(v) for t, v in measured.items()}

    # Prefer the cheapest, least contended track as the anchor; fall back to the
    # median implied anchor so one noisy node cannot skew every estimate.
    anchor = None
    if "optimized1dcnn" in measured:
        anchor = measured["optimized1dcnn"] / FALLBACK_SCALE["optimized1dcnn"]
    elif measured:
        anchor = stat.median(v / FALLBACK_SCALE[t] for t, v in measured.items())
    return measured, anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="also flag seeds trained on a different device than the majority")
    ap.add_argument("--device", default=None, choices=["hpc", "mac"],
                    help="device to estimate for (default: the one most runs used)")
    ap.add_argument("--emit-manifest", action="store_true",
                    help="write compute/remaining_runs.tsv for submit_remaining.sh")
    ap.add_argument("--cloud-grey", action="store_true",
                    help="include grey-cloud variants in the plan")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="how many jobs run in parallel (SLURM: number of GPUs you get)")
    ap.add_argument("--progress-of", metavar="TRACK:SEED",
                    help="print just the completed-epoch count for one run and exit; "
                         "used by train.sh to tell a flaky crash (progress made) from "
                         "a hard failure (none) when deciding whether to retry")
    args = ap.parse_args()

    if args.progress_of:
        tr, _, sd = args.progress_of.partition(":")
        print(ckpt_progress(tr, int(sd)) or 0)
        return

    runs = load_runs()
    devices = [r["device"] for rs in runs.values() for r in rs]
    ref_device = args.device or (stat.mode(devices) if devices else "hpc")
    measured, anchor = calibrate(runs, ref_device)

    def sec_per_epoch(track):
        predicted = anchor * FALLBACK_SCALE[track] if anchor else None
        if track in measured:
            m = measured[track]
            # A run resumed on different hardware keeps the OLD machine's per-epoch
            # times in its history, so the median can be many times too fast (a
            # cluster-trained run continued on the laptop). Anything far below what
            # this device's anchor predicts is contaminated, not genuinely quick.
            if predicted and m < 0.5 * predicted:
                return predicted, "estimated"
            return m, "measured"
        if predicted:
            return predicted, "estimated"
        return None, "unknown"

    # ---- plan ---------------------------------------------------------------
    todo, done = [], []
    for track in TRACKS:
        for seed in TARGET_SEEDS:
            status, reason, epochs_done = classify(track, seed, runs, args.strict, ref_device)
            if status == "DONE":
                done.append((track, seed, reason))
                continue
            remaining = TARGET_EPOCHS - epochs_done
            spe, src = sec_per_epoch(track)
            todo.append(dict(track=track, seed=seed, status=status, reason=reason,
                             epochs_done=epochs_done, epochs_left=remaining,
                             sec_per_epoch=spe, timing_src=src,
                             hours=(spe * remaining / 3600.0) if spe else None,
                             cloud=None))
            if args.cloud_grey:
                todo.append(dict(track=track, seed=seed, status="MISSING",
                                 reason="grey-cloud variant", epochs_done=0,
                                 epochs_left=TARGET_EPOCHS, sec_per_epoch=spe,
                                 timing_src=src,
                                 hours=(spe * TARGET_EPOCHS / 3600.0) if spe else None,
                                 cloud="grey"))

    # ---- report -------------------------------------------------------------
    print("=" * 78)
    print(f" SEED PLAN — target {len(TARGET_SEEDS)} seeds x {TARGET_EPOCHS} epochs "
          f"x {len(TRACKS)} tracks = {len(TARGET_SEEDS) * len(TRACKS)} runs")
    print(f" reference device: {ref_device}   strict={args.strict}")
    print("=" * 78)

    print(f"\nCOMPLETE ({len(done)}):")
    for track, seed, reason in done:
        print(f"  ok   {track:16s} seed {seed}   {reason}")

    print(f"\nTO RUN ({len(todo)}):")
    print(f"  {'track':16s} {'sd':2s} {'variant':7s} {'status':10s} {'ep':>3s} "
          f"{'s/ep':>7s} {'hours':>6s}  reason")
    for r in todo:
        var = r["cloud"] or "base"
        spe = f"{r['sec_per_epoch']:.0f}" if r["sec_per_epoch"] else "?"
        hrs = f"{r['hours']:.1f}" if r["hours"] else "?"
        mark = "~" if r["timing_src"] == "estimated" else " "
        print(f"  {r['track']:16s} {r['seed']:<2d} {var:7s} {r['status']:10s} "
              f"{r['epochs_left']:3d} {spe:>6s}{mark} {hrs:>6s}  {r['reason']}")

    total = sum(r["hours"] for r in todo if r["hours"])
    unknown = [r for r in todo if not r["hours"]]
    print("\n" + "-" * 78)
    print(f"  serial wall clock      : {total:7.1f} h  ({total / 24:.1f} days)")
    if args.concurrency > 1:
        # Longest-job bound: perfect packing can never beat the slowest single job.
        longest = max((r["hours"] for r in todo if r["hours"]), default=0)
        packed = max(total / args.concurrency, longest)
        print(f"  with {args.concurrency} concurrent jobs : {packed:7.1f} h  "
              f"({packed / 24:.1f} days)   [longest single job {longest:.1f} h]")
    if unknown:
        print(f"  WARNING: {len(unknown)} run(s) have no timing basis")
    print(f"  '~' = extrapolated from the {ref_device} anchor, not measured on this track.")
    print(f"  Excludes queue wait, data loading on first epoch, and evaluation.")
    print("-" * 78)

    # ---- provenance table for the paper -------------------------------------
    print("\nTRAINING PROVENANCE (existing runs — for the paper's reproducibility note):")
    print(f"  {'track':16s} {'sd':2s} {'ep':>3s} {'device':7s} {'s/ep':>7s} {'wall_h':>7s}  objective")
    for (track, seed, cloud), rs in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                        kv[0][2] or "")):
        for r in rs:
            if cloud is not None:
                continue
            obj = "cf" if r["cf_train"] else ("cf-invariance" if r["cf_invariance"] else "plain")
            ok = "" if objective_matches(track, r) else "  <-- MISMATCH"
            spe = f"{r['sec_median']:.0f}" if r["sec_median"] else "?"
            wh = f"{r['wall_hours']:.2f}" if r["wall_hours"] else "?"
            print(f"  {track:16s} {seed:<2d} {r['epochs']:3d} {r['device']:7s} "
                  f"{spe:>7s} {wh:>7s}  {obj}{ok}")

    if args.emit_manifest:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remaining_runs.tsv")
        with open(path, "w") as fh:
            fh.write("# track\tseed\tepochs\tcloud\tflags\test_hours\tstatus\n")
            for r in todo:
                flags = " ".join(OBJECTIVE[r["track"]]) or "-"
                fh.write(f"{r['track']}\t{r['seed']}\t{TARGET_EPOCHS}\t"
                         f"{r['cloud'] or '-'}\t{flags}\t"
                         f"{r['hours'] or 0:.2f}\t{r['status']}\n")
        print(f"\nmanifest -> {path}  ({len(todo)} runs)")
        print("submit with:  bash compute/submit_remaining.sh")


if __name__ == "__main__":
    main()
