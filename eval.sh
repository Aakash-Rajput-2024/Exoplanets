#!/usr/bin/env bash
# =============================================================================
# eval.sh — reproducible full re-evaluation of every trained v2 checkpoint.
#
# Scores each HPC-trained checkpoint through the unified A–J evaluation pipeline
# (src/evaluation) and builds a cross-track comparison. It ONLY evaluates existing
# checkpoints — it never trains or mutates weights/data.
#
#   Sections: A in-dist per-gas + R²-vs-exposure sweep · B classical baselines ·
#             C cross-generator (pRT/TauREx) · D PSG anchor · E solar-system (real GT) ·
#             F real Earth (real GT) · G transiting OOD probe · H published retrieval ·
#             I calibration (SBC/TARP/PIT/ECE) · J OOD honesty
#
# Usage:
#   ./eval.sh                 # full matrix, all sections
#   NCAL=1000 TMC=30 ./eval.sh   # heavier calibration (slower, esp. transformer)
#   ONLY="optimized1dcnn transformerarch" ./eval.sh   # subset of tracks
#   SUITES="A B E F" ./eval.sh   # only some sections (fast core)
#
# Outputs:
#   src/evaluation/results/<track>/report.md          per-track full report
#   src/evaluation/results/<track>/summary.json       machine roll-up
#   src/evaluation/results/<track>/<suite>/result.json per-section detail (+figs)
#   src/evaluation/results/leaderboard.json           accumulated headlines
#   src/evaluation/results/COMPARISON.md / .json      side-by-side cross-track
#   logs/eval_<label>_<ts>.log                        full stdout per track
# =============================================================================
set -o pipefail   # NOT -u: macOS bash 3.2 treats empty-array expansion as unbound

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/src"
PY="${PY:-python3}"

# ---- tunables (env-overridable) --------------------------------------------
SUITES="${SUITES:-all}"                 # sections to run: 'all' or e.g. "A B E F I"
NCAL="${NCAL:-500}"                     # calibration: planets sampled (section I)
TMC="${TMC:-20}"                        # calibration: MC-dropout draws/seed (section I)
ALPHAS="${ALPHAS:-0.3 1 3 10 30 100 300}"   # exposure sweep points (α=√(t/t_nom))
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

# ---- the evaluation matrix -------------------------------------------------
# One row per checkpoint set: "label | track | seeds | suffix | note".
# Seed lists are EXPLICIT and per-checkpoint (the causal dir mixes architectures
# under similar filenames — causal_cfi is the small transformer at seeds 0,1 only;
# causal_xl is the 2M-param NasaInaraCausalNet at seeds 0,1,2 — do not widen these).
# Fast CNN tracks are front-loaded so partial completion still yields core results.
MATRIX=(
  "optimized1dcnn        | optimized1dcnn  | 0 1 2 |        | matched CNN (BN+progressive kernels), 3 seeds"
  "optimized1dcnn_grey   | optimized1dcnn  | 0     | _grey  | cloud-robust CNN (grey-cloud augmentation, v2)"
  "original1dcnn         | original1dcnn   | 0     |        | paper-baseline CNN — only 10 epochs (undertrained; reference, not matched)"
  "transformerarch       | transformerarch | 0     |        | matched transformer backbone (48 ep)"
  "causal                | causal          | 0     |        | transformer + exact do(env) counterfactual AUGMENTATION (--cf)"
  "causal_cfi            | causal_cfi      | 0 1   | _cfi   | transformer + counterfactual-INVARIANCE objective (seed1 short: 13 ep)"
  "causal_xl             | causal_xl       | 0 1 2 | _cfi   | 2M-param do-calculus net + counterfactual-invariance, 3 seeds"
)

# allow ONLY="track1 track2" to restrict
run_one () {
  local label="$1" track="$2" seeds="$3" suffix="$4"
  local logf="logs/eval_${label}_${TS}.log"
  local sfx_args=()
  [ -n "$suffix" ] && sfx_args=(--suffix "$suffix")
  echo ">>> [$label]  track=$track seeds=[$seeds] suffix='${suffix:-}'  suites=$SUITES"
  echo "    log: $logf"
  # shellcheck disable=SC2086
  "$PY" -m evaluation.run_eval "$track" \
      --seeds $seeds ${sfx_args[@]+"${sfx_args[@]}"} \
      --suites $SUITES \
      --alphas $ALPHAS \
      --n-cal "$NCAL" --T "$TMC" \
      >"$logf" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "    OK"; else echo "    FAILED (rc=$rc) — see $logf"; fi
  return $rc
}

echo "=============================================================="
echo " eval.sh — full re-evaluation   ($(date))"
echo " suites=$SUITES  n-cal=$NCAL  T=$TMC  alphas=[$ALPHAS]"
echo "=============================================================="
declare -a OK_LABELS FAIL_LABELS
for row in "${MATRIX[@]}"; do
  IFS='|' read -r label track seeds suffix note <<<"$row"
  label="$(echo "$label" | xargs)"; track="$(echo "$track" | xargs)"
  seeds="$(echo "$seeds" | xargs)"; suffix="$(echo "$suffix" | xargs)"
  if [ -n "${ONLY:-}" ] && ! grep -qw "$track" <<<"$ONLY" && ! grep -qw "$label" <<<"$ONLY"; then
    echo "--- skip $label (not in ONLY) ---"; continue
  fi
  if run_one "$label" "$track" "$seeds" "$suffix"; then
    OK_LABELS+=("$label"); else FAIL_LABELS+=("$label"); fi
done

echo
echo "=== building cross-track comparison (COMPARISON.md) ==="
"$PY" -m evaluation.aggregate_report >"logs/aggregate_${TS}.log" 2>&1 \
  && echo "    OK -> src/evaluation/results/COMPARISON.md" \
  || echo "    aggregate FAILED — see logs/aggregate_${TS}.log"

echo
echo "=============================================================="
echo " DONE.  ok=[${OK_LABELS[*]:-}]  failed=[${FAIL_LABELS[*]:-}]"
echo " reports: src/evaluation/results/<track>/report.md"
echo " compare: src/evaluation/results/COMPARISON.md"
echo "=============================================================="
