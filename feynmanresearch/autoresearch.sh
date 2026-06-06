#!/usr/bin/env bash
set -euo pipefail

cd /Users/aakashrajput/MachineLearning/Exoplanets
source /Users/aakashrajput/MachineLearning/venvNLP/bin/activate

mkdir -p feynmanresearch/logs

ts=$(date +%Y%m%d_%H%M%S)
log="feynmanresearch/logs/baseline_test_${ts}.log"

echo "Running baseline transformer evaluation..."
echo "Log: ${log}"
python feynmanresearch/run_transformer_test_cpu.py 2>&1 | tee "${log}"
