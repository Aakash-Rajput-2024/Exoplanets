#!/usr/bin/env bash
# =============================================================================
# ULTRAPLAN v2 — retrain + evaluate the INARA retrieval models.
#
# Runs the chosen architecture(s) under ONE matched budget (fixes C4), on the
# physical contrast observable with injected noise (C1), CLR labels (C2), per-λ
# asinh encoding (H3), on the frozen 80/10/10 planet-ID split (C3), across seeds
# (H6), with provenance-stamped checkpoints (C6). Then evaluates on the held-out
# TEST set (log-space metrics + bootstrap CIs), runs the SNR sweep (C1 figure)
# and the held-out cloud-family transfer test (C5), and collates a cross-track
# table. Every stage is tee'd to logs/run_<timestamp>/.
#
# INTERACTIVE: run `./run.sh` and it asks which track(s), how many epochs (20),
# how many seeds, and whether to add grey-cloud variants. Press Enter to accept
# the [defaults].
#
# NON-INTERACTIVE (scripting): set NONINTERACTIVE=1 and use env vars, e.g.
#   NONINTERACTIVE=1 TRACKS="transformerarch" EPOCHS=20 SEEDS="0" ./run.sh
#
# The 94 GB→cache build is a SEPARATE one-time step (not re-run here):
#   PYTHONPATH=src python -m common.build_cache
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src"
PY="${PY:-python}"

ALL_TRACKS="original1dcnn 2channel1dcnn optimized1dcnn transformerarch causal"

# ---- defaults (each is an interactive prompt default AND an env override) -----
# EPOCHS defaults to the registry MATCHED budget (50). A shorter run is a
# PRELIMINARY smoke; the evaluator stamps epochs_run and flags undertrained runs.
TRACKS="${TRACKS:-$ALL_TRACKS}"
EPOCHS="${EPOCHS:-50}"
SEEDS="${SEEDS:-0 1 2}"
RUN_CLOUD="${RUN_CLOUD:-0}"
CACHE="${CACHE:-$REPO/data/cache_v2}"

# ---- interactive selection ---------------------------------------------------
if [ "${NONINTERACTIVE:-0}" != "1" ]; then
  echo "=================================================================="
  echo " ULTRAPLAN v2 — training launcher"
  echo "=================================================================="
  echo "Which track(s) to train?  (space-separated for several, e.g. '3 4 5')"
  echo "   0) all five"
  echo "   1) original1dcnn      (paper-baseline CNN)"
  echo "   2) 2channel1dcnn      (same arch as original)"
  echo "   3) optimized1dcnn     (BatchNorm + AdaptiveAvgPool CNN)"
  echo "   4) transformerarch    (CNN→Transformer→GAP)"
  echo "   5) causal             (transformer backbone; DSCM base)"
  read -rp "Enter number(s) [0]: " tc
  tc="${tc:-0}"
  if [[ " $tc " == *" 0 "* ]]; then
    TRACKS="$ALL_TRACKS"
  else
    sel=""
    for n in $tc; do
      case "$n" in
        1) sel="$sel original1dcnn" ;;
        2) sel="$sel 2channel1dcnn" ;;
        3) sel="$sel optimized1dcnn" ;;
        4) sel="$sel transformerarch" ;;
        5) sel="$sel causal" ;;
        *) echo "invalid choice '$n'"; exit 1 ;;
      esac
    done
    TRACKS="$(echo $sel | xargs)"
  fi

  read -rp "How many epochs? [${EPOCHS}]: " ep
  EPOCHS="${ep:-$EPOCHS}"

  read -rp "Which seed(s)? (space-separated; ≥3 for the H6 CI study) [${SEEDS}]: " sd
  SEEDS="${sd:-$SEEDS}"

  read -rp "Also train grey-cloud variants for the C5 seen/unseen gap? (y/N): " rc
  case "${rc:-n}" in y|Y|yes|YES) RUN_CLOUD=1 ;; *) RUN_CLOUD=0 ;; esac
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$REPO/logs/run_$TS"
mkdir -p "$LOG"
exec > >(tee "$LOG/run.log") 2>&1

echo "=================================================================="
echo " ULTRAPLAN v2 pipeline   $TS"
echo " tracks : $TRACKS"
echo " seeds  : $SEEDS   epochs: $EPOCHS   grey-cloud variants: $RUN_CLOUD"
echo " cache  : $CACHE"
echo " logs   : $LOG"
echo "=================================================================="

# ---- Stage 0: environment + cache preflight ----------------------------------
$PY -c "import torch,sys;print('python',sys.version.split()[0],'| torch',torch.__version__,'| mps',torch.backends.mps.is_available(),'| cuda',torch.cuda.is_available())"
if [ ! -f "$CACHE/manifest.json" ]; then
  echo "ERROR: $CACHE/manifest.json not found."
  echo "Build the unified cache first (one-time, reads the 94 GB dataset):"
  echo "    PYTHONPATH=src $PY -m common.build_cache"
  exit 1
fi
echo "cache_v2 manifest: $(grep -o '\"config_hash\": \"[^\"]*\"' "$CACHE/manifest.json")"

# ---- Stage 1: train (matched budget, per seed) -------------------------------
echo; echo "########## STAGE 1: TRAIN ##########"
for t in $TRACKS; do
  for s in $SEEDS; do
    echo; echo "--- train $t seed $s (clean+noise) ---"
    if ! $PY -m common.train_runner "$t" --seed "$s" --epochs "$EPOCHS" --cache "$CACHE" \
        2>&1 | tee "$LOG/train_${t}_seed${s}.log"; then
      echo "WARN: train $t seed $s FAILED — see $LOG/train_${t}_seed${s}.log; continuing"
    fi
    if [ "$RUN_CLOUD" = "1" ]; then
      echo "--- train $t seed $s (grey-cloud, C5) ---"
      if ! $PY -m common.train_runner "$t" --seed "$s" --epochs "$EPOCHS" --cloud grey --cache "$CACHE" \
          2>&1 | tee "$LOG/train_${t}_seed${s}_grey.log"; then
        echo "WARN: grey-cloud train $t seed $s FAILED; continuing"
      fi
    fi
  done
done

# ---- Stage 2: evaluate on TEST + SNR sweep -----------------------------------
echo; echo "########## STAGE 2: EVALUATE (test set + SNR sweep) ##########"
for t in $TRACKS; do
  $PY -m common.evaluate "$t" --seeds $SEEDS --cache "$CACHE" 2>&1 | tee "$LOG/eval_${t}.log" \
    || echo "WARN: eval $t FAILED (no checkpoints?); continuing"
done

# ---- Stage 2b: input-leakage regression probe (must PASS) --------------------
echo; echo "########## STAGE 2b: LEAKAGE PROBE (contrast vs SNR channel) ##########"
for t in $TRACKS; do
  # non-fatal to the pipeline, but a FAIL means the SNR channel is carrying the
  # retrieval signal — investigate before trusting any number for that track.
  $PY -m common.leakage_probe "$t" --cache "$CACHE" 2>&1 | tee "$LOG/leakage_${t}.log" \
    || echo "WARN: LEAKAGE PROBE FAILED for $t — SNR channel may be leaking; see $LOG/leakage_${t}.log"
done

# ---- Stage 3: held-out cloud-family transfer (C5) ----------------------------
echo; echo "########## STAGE 3: CLOUD-FAMILY TRANSFER (C5) ##########"
for t in $TRACKS; do
  $PY -m common.eval_cloud_transfer "$t" --seeds $SEEDS --cache "$CACHE" \
      2>&1 | tee "$LOG/cloudtransfer_${t}.log" || echo "WARN: cloud-transfer $t FAILED; continuing"
  if [ "$RUN_CLOUD" = "1" ]; then
    $PY -m common.eval_cloud_transfer "$t" --seeds $SEEDS --suffix _grey --cache "$CACHE" \
        2>&1 | tee "$LOG/cloudtransfer_${t}_grey.log" || echo "WARN: cloud-transfer $t _grey FAILED; continuing"
  fi
done

# ---- Stage 4: cross-track architecture comparison ----------------------------
echo; echo "########## STAGE 4: ARCHITECTURE COMPARISON (C4) ##########"
$PY -m common.summarize 2>&1 | tee "$LOG/summary.log"

echo; echo "DONE. All logs, per-epoch CSV/JSON, details, and figures under:"
echo "  $LOG"
echo "  src/<track>/{logs_v2,charts_v2,eval_v2}/   and   results_v2/"
