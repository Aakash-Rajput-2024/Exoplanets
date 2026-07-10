#!/bin/bash
# =============================================================================
# PARAM Rudra — Evaluate all trained models after training completes.
#
# Submit this AFTER all training jobs finish (check with squeue).
# Runs: eval (test+SNR sweep), leakage probe, cloud-family transfer, summary.
#
# Usage:
#   cd /scratch/aakashr_iitp/Exoplanets
#   sbatch cluster/submit_eval.sh
# =============================================================================
#SBATCH --job-name=exo_eval_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs_slurm/eval_all_%J.out
#SBATCH --error=logs_slurm/eval_all_%J.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load MLDL/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate exoplanet_env

export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TRACKS="original1dcnn optimized1dcnn transformerarch causal causal_cfi"
SEEDS="0 1 2"
CACHE="$SLURM_SUBMIT_DIR/data/cache_v2"

echo "=== Eval all | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Stage 1: Evaluate on test set + SNR sweep
echo; echo "########## EVALUATE (test + SNR sweep) ##########"
for t in $TRACKS; do
    echo "--- eval $t ---"
    python -m common.evaluate "$t" --seeds $SEEDS --cache "$CACHE" 2>&1 || \
        echo "WARN: eval $t FAILED; continuing"
done

# Stage 2: Leakage probe (must pass for the SNR channel to be trustworthy)
echo; echo "########## LEAKAGE PROBE ##########"
for t in $TRACKS; do
    python -m common.leakage_probe "$t" --cache "$CACHE" 2>&1 || \
        echo "WARN: LEAKAGE PROBE FAILED for $t"
done

# Stage 3: Cloud-family transfer (C5)
echo; echo "########## CLOUD-FAMILY TRANSFER ##########"
for t in $TRACKS; do
    python -m common.eval_cloud_transfer "$t" --seeds $SEEDS --cache "$CACHE" 2>&1 || \
        echo "WARN: cloud-transfer $t FAILED; continuing"
done

# Stage 4: Cross-track summary table
echo; echo "########## ARCHITECTURE COMPARISON ##########"
python -m common.summarize 2>&1 || echo "WARN: summarize FAILED"

echo; echo "=== done eval all | $(date) ==="
