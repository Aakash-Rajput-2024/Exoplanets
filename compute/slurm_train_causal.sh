#!/bin/bash
# ============================================================================
# PARAM Rudra — train the ~2M-param do-calculus model (causal_xl) with the
# exact-intervention counterfactual-invariance objective.
#
# Submit all 3 seeds as an array job:   sbatch slurm_train_causal.sh
# Submit a single seed:                 sbatch --array=0 slurm_train_causal.sh
# ============================================================================
#SBATCH --job-name=causal_xl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-2                     # one job per seed (0,1,2)
#SBATCH --output=logs_slurm/causal_xl_seed%a_%A.out
#SBATCH --error=logs_slurm/causal_xl_seed%a_%A.err

set -euo pipefail
mkdir -p logs_slurm
cd "$SLURM_SUBMIT_DIR"

# --- environment ---
module purge
module load mldl/Miniconda
# In a non-interactive batch shell `conda activate` needs conda sourced first.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate exoplanet_env

# Our modules live in ./src and are run as `python -m common.<...>`
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
echo "=== causal_xl | seed $SEED | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# --resume writes an atomic per-epoch snapshot (checkpoints_v2/last_seed${SEED}_cfi.pth)
# so a preempted/killed job continues where it stopped; the best-so-far checkpoint
# (model_best_seed${SEED}_cfi.pth) is (re)written every epoch that improves val loss.
python -m common.train_runner causal_xl --cf-invariance --seed "$SEED" --resume --diag

echo "=== done seed $SEED | $(date) ==="
