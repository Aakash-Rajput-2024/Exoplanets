#!/usr/bin/env bash
# =============================================================================
# Finish the REMAINING seed-0 tracks at 10 epochs each (preliminary), then
# evaluate all five + summarize. Every training call is:
#   * caffeinate -dims  -> machine cannot sleep mid-run (this is what killed
#                          transformerarch-grey at epoch 1, twice)
#   * --resume          -> continues from last_seed0*.pth if interrupted
#   * --diag            -> per-alpha val R2 logged each epoch
# NOTE: 10 epochs with warmup_epochs=5 is HALF warmup -> these are quick
# PRELIMINARY fills for the comparison table, not budget-matched final numbers.
# set -e is NOT used, so one failed job never aborts the rest.
# =============================================================================
set -uo pipefail
REPO="/Users/aakashrajput/MachineLearning/Exoplanets"
cd "$REPO"; export PYTHONPATH="$REPO/src"
PY="/Users/aakashrajput/MachineLearning/venvNPT/bin/python"
CACHE="$REPO/data/cache_v2"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$REPO/logs/finish10_$TS"; mkdir -p "$LOG"
echo "=== finish_10ep  $TS  logs:$LOG ==="

run() { local n="$1"; shift; echo; echo "##### $n $(date) #####"
  if caffeinate -dims "$@" 2>&1 | tee "$LOG/$n.log"; then echo "##### DONE $n #####"
  else echo "##### WARN $n FAILED (continuing) #####"; fi; }

# ---- Stage 1: train the remaining tracks, 10 epochs each --------------------
# original1dcnn + 2channel1dcnn: never trained in v2 (fresh 10-ep runs).
run train_original1dcnn  "$PY" -m common.train_runner original1dcnn --seed 0 --epochs 10 --resume --diag --cache "$CACHE"
run train_2channel1dcnn  "$PY" -m common.train_runner 2channel1dcnn --seed 0 --epochs 10 --resume --diag --cache "$CACHE"
# transformerarch grey: was stuck at epoch 1 -> resume to 10 total.
run train_transformer_grey "$PY" -m common.train_runner transformerarch --seed 0 --epochs 10 --cloud grey --resume --diag --cache "$CACHE"

# ---- Stage 2: evaluate every track (incl. the unscored causal) + summarize --
for t in original1dcnn 2channel1dcnn optimized1dcnn transformerarch causal; do
  run "eval_$t" "$PY" -m common.evaluate "$t" --seeds 0 --cache "$CACHE"
done
run probe_causal "$PY" -m common.leakage_probe causal --seed 0 --cache "$CACHE"
run summarize    "$PY" -m common.summarize

echo; echo "=== finish_10ep COMPLETE $(date) — see results_v2/architecture_comparison.txt ==="
