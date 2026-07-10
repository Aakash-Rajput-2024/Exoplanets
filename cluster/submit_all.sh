#!/bin/bash
# =============================================================================
# PARAM Rudra — Train ALL v2 models (5 tracks × 3 seeds = 15 jobs).
#
# This master SLURM script submits one job per (track, seed) combination.
# Each job trains under the identical matched budget (C4) on a single GPU.
#
# Usage from the login node:
#   cd /scratch/aakashr_iitp/Exoplanets
#   bash cluster/submit_all.sh              # submit all 15 jobs
#   bash cluster/submit_all.sh --dry-run    # just print what would run
#   bash cluster/submit_all.sh --only transformerarch  # single track, 3 seeds
#
# Tracks (from registry.py):
#   original1dcnn    — Paper-baseline 4-conv CNN (36M params)
#   optimized1dcnn   — CNN + BatchNorm + progressive kernels
#   transformerarch  — CNN ×8 downsample → 2-layer Transformer → GAP → MLP
#   causal           — Transformer + do(env) counterfactual augmentation (--cf)
#   causal_cfi       — Transformer + counterfactual-invariance objective (--cf-invariance)
#
# Optional: add --cloud-grey to also train grey-cloud variants (C5 study).
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Configurable
TRACKS="${ONLY:-causal_cfi causal transformerarch optimized1dcnn original1dcnn}"
SEEDS="0 1 2"
EPOCHS=50
CLOUD_GREY=0
DRY_RUN=0

# Parse args
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=1 ;;
        --cloud-grey)  CLOUD_GREY=1 ;;
        --only)        shift; TRACKS="$1" ;;
    esac
done

# Handle --only with next arg
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) TRACKS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p logs_slurm

echo "============================================================"
echo " ULTRAPLAN v2 — Cluster Training Submission"
echo " Tracks: $TRACKS"
echo " Seeds:  $SEEDS    Epochs: $EPOCHS"
echo " Grey-cloud variants: $CLOUD_GREY"
echo "============================================================"

for TRACK in $TRACKS; do
    for SEED in $SEEDS; do
        JOB_NAME="${TRACK}_s${SEED}"

        # Build the train command
        TRAIN_CMD="python -m common.train_runner ${TRACK} --seed ${SEED} --epochs ${EPOCHS} --resume --diag"

        # Add causal flags based on track name
        case "$TRACK" in
            causal)     TRAIN_CMD="$TRAIN_CMD --cf" ;;
            causal_cfi) TRAIN_CMD="$TRAIN_CMD --cf-invariance" ;;
            causal_xl)  TRAIN_CMD="$TRAIN_CMD --cf-invariance" ;;
        esac

        if [ "$DRY_RUN" = "1" ]; then
            echo "[dry-run] would submit: $JOB_NAME → $TRAIN_CMD"
            continue
        fi

        echo "Submitting $JOB_NAME..."
        sbatch \
            --job-name="$JOB_NAME" \
            --partition=gpu \
            --nodes=1 \
            --gres=gpu:1 \
            --ntasks-per-node=1 \
            --cpus-per-task=8 \
            --time=12:00:00 \
            --output="logs_slurm/${JOB_NAME}_%J.out" \
            --error="logs_slurm/${JOB_NAME}_%J.err" \
            --wrap="
set -eo pipefail
cd \$SLURM_SUBMIT_DIR

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate exoplanet_env

export PYTHONPATH=\"\$SLURM_SUBMIT_DIR/src\"
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}

echo '=== ${JOB_NAME} | \$(hostname) | \$(date) ==='
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c \"import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())\"

${TRAIN_CMD}

echo '=== done ${JOB_NAME} | \$(date) ==='
"
    done

    # Optional grey-cloud variant
    if [ "$CLOUD_GREY" = "1" ]; then
        for SEED in $SEEDS; do
            JOB_NAME="${TRACK}_grey_s${SEED}"
            echo "Submitting $JOB_NAME (grey-cloud)..."
            sbatch \
                --job-name="$JOB_NAME" \
                --partition=gpu \
                --nodes=1 \
                --gres=gpu:1 \
                --ntasks-per-node=1 \
                --cpus-per-task=8 \
                --time=12:00:00 \
                --output="logs_slurm/${JOB_NAME}_%J.out" \
                --error="logs_slurm/${JOB_NAME}_%J.err" \
                --wrap="
set -eo pipefail
cd \$SLURM_SUBMIT_DIR

module purge
module load MLDL/miniconda3
source /home/apps/MLDL/DL-CondaPy3/etc/profile.d/conda.sh
conda activate exoplanet_env

export PYTHONPATH=\"\$SLURM_SUBMIT_DIR/src\"
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}

echo '=== ${JOB_NAME} | \$(hostname) | \$(date) ==='
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m common.train_runner ${TRACK} --seed ${SEED} --epochs ${EPOCHS} --cloud grey --resume --diag

echo '=== done ${JOB_NAME} | \$(date) ==='
"
        done
    fi
done

echo
echo "All jobs submitted. Monitor with:"
echo "  squeue -u aakashr_iitp"
echo "  tail -f logs_slurm/<job_name>_<jobid>.out"
