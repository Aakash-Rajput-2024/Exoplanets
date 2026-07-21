#!/bin/bash
# =============================================================================
# PARAM Rudra — one-time environment setup.
#
# Run this INSIDE an interactive GPU allocation:
#   salloc --nodes=1 --gres=gpu:1 --time=00:30:00 --partition=gpu
#   ssh <assigned_node>       # e.g. ssh ragpu003
#   bash compute/setup_env.sh
#   exit; exit
# =============================================================================
set -euo pipefail

echo "=== Setting up exoplanet_env on $(hostname) ==="

module load MLDL/miniconda3 2>/dev/null || true
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if it doesn't exist
if conda info --envs | grep -q exoplanet_env; then
    echo "Environment 'exoplanet_env' already exists — updating."
    conda activate exoplanet_env
else
    echo "Creating conda environment 'exoplanet_env'..."
    conda create --name exoplanet_env python=3.10 -y
    conda activate exoplanet_env
fi

# Install PyTorch with CUDA (cluster's CUDA version)
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining deps (pip, NOT torch — keep conda's CUDA build)
echo "Installing pip dependencies..."
pip install -r requirements_cluster.txt

# Verify
echo
echo "=== Verification ==="
python -c "
import torch, sys
print(f'Python:  {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"})')
print(f'Devices: {torch.cuda.device_count()}')
"

echo
echo "=== Environment ready. Exit the compute node (exit; exit) ==="
