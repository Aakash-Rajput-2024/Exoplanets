#!/bin/bash
# Full cross-generator evaluation run: generate both caches (N=2000) with TauREx,
# then run all four models' crosstest. Generation uses .venv (TauREx), evaluation
# uses venvNLP (torch).
set -e
REPO="/Users/aakashrajput/MachineLearning/Exoplanets"
VENV="$REPO/MultiREx-public/.venv/bin/python"
TORCH="/Users/aakashrajput/MachineLearning/venvNLP/bin/python"
N=2000

echo "=== [1/6] generate planet cache (N=$N) ==="
$VENV "$REPO/src/evaluation/crossgen/build_eval_cache.py" --engine taurex \
    --feature-mode planet --n $N --out "$REPO/data/cache_crossgen_taurex_planet" \
    2>&1 | grep -vE "^(Numba|INFO|WARNING|taurex|Loading)"

echo "=== [2/6] generate both cache (N=$N) ==="
$VENV "$REPO/src/evaluation/crossgen/build_eval_cache.py" --engine taurex \
    --feature-mode both --n $N --out "$REPO/data/cache_crossgen_taurex_both" \
    2>&1 | grep -vE "^(Numba|INFO|WARNING|taurex|Loading)"

echo "=== [3/6] transformer crosstest ==="
$TORCH "$REPO/src/models/transformerarch/crosstest.py" --engine taurex 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"
echo "=== [4/6] original1dcnn crosstest ==="
$TORCH "$REPO/src/models/original1dcnn/crosstest.py" --engine taurex 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"
echo "=== [5/6] causal crosstest ==="
$TORCH "$REPO/src/models/causal/cnn_trnas/crosstest.py" --engine taurex 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"
echo "=== [6/6] 2channel crosstest ==="
$TORCH "$REPO/src/2channel1dcnn/crosstest.py" --engine taurex 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"

echo "=== DONE ==="
