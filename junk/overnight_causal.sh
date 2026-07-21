#!/usr/bin/env bash
# =============================================================================
# Overnight run: causal (do-calculus) counterfactual track + the missing
# transformer grey-cloud variant, then evaluate. Every job is:
#   * RESUMABLE   (--resume: writes checkpoints_v2/last_seed0*.pth each epoch;
#                  just re-run this script to continue after a crash/sleep)
#   * LOGGED      (per-job tee into logs/overnight_<ts>/)
#   * DIAGNOSED   (--diag: per-alpha val R2 logged each epoch)
# set -e is deliberately NOT used, so one failed job never aborts the rest.
# Launched detached via nohup, so it survives the terminal closing.
# =============================================================================
set -uo pipefail

REPO="/Users/aakashrajput/MachineLearning/Exoplanets"
cd "$REPO"
export PYTHONPATH="$REPO/src"
PY="/Users/aakashrajput/MachineLearning/venvNPT/bin/python"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/logs/overnight_$TS"
mkdir -p "$LOG"
echo "==================================================================="
echo " OVERNIGHT causal-CF run   $TS"
echo " logs: $LOG"
echo " python: $PY"
echo "==================================================================="

run() {  # run() <name> <cmd...>  — labeled, tee'd, non-fatal
  local name="$1"; shift
  echo; echo "########## START $name  $(date) ##########"
  if "$@" 2>&1 | tee "$LOG/$name.log"; then
    echo "########## DONE  $name  $(date) ##########"
  else
    echo "########## WARN  $name FAILED (continuing)  $(date) ##########"
  fi
}

# ---- Stage 1: train ---------------------------------------------------------
# 1) Causal do-calculus track — exact environment counterfactuals (same
#    atmosphere, re-paired host star + noise). Compare against the already-done
#    transformerarch clean model (identical architecture, no CF) = the control.
run train_causal_cf \
  "$PY" -m common.train_runner causal --seed 0 --epochs 50 --cf --resume --diag --cache "$REPO/data/cache_v2"

# 2) Transformer grey-cloud (only 1 epoch had completed before the freeze).
run train_transformer_grey \
  "$PY" -m common.train_runner transformerarch --seed 0 --epochs 50 --cloud grey --resume --diag --cache "$REPO/data/cache_v2"

# ---- Stage 2: evaluate whatever finished ------------------------------------
run eval_causal        "$PY" -m common.evaluate      causal          --seeds 0 --cache "$REPO/data/cache_v2"
run eval_transformer   "$PY" -m common.evaluate      transformerarch --seeds 0 --cache "$REPO/data/cache_v2"
run probe_causal       "$PY" -m common.leakage_probe causal          --seed 0  --cache "$REPO/data/cache_v2"
run summarize          "$PY" -m common.summarize

echo; echo "===================================================================",
echo " OVERNIGHT RUN COMPLETE  $(date)"
echo " results: src/models/causal/cnn_trnas/eval_v2/  src/models/transformerarch/eval_v2/  results_v2/"
echo "==================================================================="
