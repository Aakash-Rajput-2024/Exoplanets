#!/bin/bash
# Full cross-generator run with petitRADTRANS (reflected-light + scattering).
# Generation uses .venv_prt; evaluation uses venvNLP (torch).
#
# Robustness notes (why this script looks the way it does):
#  - Generation steps read stdin from /dev/null. If pRT ever fails to find a
#    c-k opacity file it drops into an interactive Keeper download prompt that
#    blocks on stdin forever (this is what stalled the Jun-23 run at step 1/6
#    and left a 49-byte log). With stdin closed it EOFs and errors instead.
#  - Each step's FULL raw output is tee'd to prt_logs/<step>.log, so a failure
#    leaves a trace even though the console only shows the filtered summary.
#  - We capture each step's real exit code (PIPESTATUS) and abort loudly on
#    failure rather than letting a filtering grep mask it.
set -uo pipefail
REPO="/Users/aakashrajput/MachineLearning/Exoplanets"
PRT="$REPO/.venv_prt/bin/python"
TORCH="/Users/aakashrajput/MachineLearning/venvNLP/bin/python"
N="${N:-2000}"                       # override with:  N=200 bash run_full_prt.sh
LOGDIR="$REPO/src/evaluation/crossgen/prt_logs"
mkdir -p "$LOGDIR"
F='Chrome|Rendering|Ending|webpage|Starting|Loading|loaded|Warning|warn|deprecat|FutureWarning'

gen () {  # $1=feature-mode  $2=out-dir
  local mode="$1" out="$2" log="$LOGDIR/gen_${1}.log"
  echo "=== generate prt $mode cache (N=$N)  ->  $log ==="
  $PRT "$REPO/src/evaluation/crossgen/build_eval_cache.py" --engine prt \
      --feature-mode "$mode" --n "$N" --out "$out" </dev/null 2>&1 | tee "$log" \
      | grep -avE "$F" | grep -aE 'grid|generated|calibration|wrote'
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "!! generation ($mode) FAILED rc=$rc — tail of $log:"; tail -20 "$log"; exit "$rc"; }
}

crosstest () {  # $1=script  $2=label
  local script="$1" label="$2" log="$LOGDIR/crosstest_${2}.log"
  echo "=== $label crosstest (prt)  ->  $log ==="
  $TORCH "$script" --engine prt </dev/null 2>&1 | tee "$log" \
      | grep -E "HEADLINE|Avg|wrote.*details"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "!! $label crosstest FAILED rc=$rc — tail of $log:"; tail -20 "$log"; exit "$rc"; }
}

echo "=== [1/6] ==="; gen planet "$REPO/data/cache_crossgen_prt_planet"
echo "=== [2/6] ==="; gen both   "$REPO/data/cache_crossgen_prt_both"
echo "=== [3/6] ==="; crosstest "$REPO/src/models/transformerarch/crosstest.py"   transformer
echo "=== [4/6] ==="; crosstest "$REPO/src/models/original1dcnn/crosstest.py"     original1dcnn
echo "=== [5/6] ==="; crosstest "$REPO/src/models/causal/cnn_trnas/crosstest.py"  causal
echo "=== [6/6] ==="; crosstest "$REPO/src/2channel1dcnn/crosstest.py"     2channel

echo "=== DONE ==="
