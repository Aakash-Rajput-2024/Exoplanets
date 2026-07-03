#!/bin/bash
# Full cross-generator run with petitRADTRANS (reflected-light + scattering).
# Generation uses .venv_prt; evaluation uses venvNLP (torch).
set -e
REPO="/Users/aakashrajput/MachineLearning/Exoplanets"
PRT="$REPO/.venv_prt/bin/python"
TORCH="/Users/aakashrajput/MachineLearning/venvNLP/bin/python"
N=2000
F='Chrome|Rendering|Ending|webpage|Starting|Loading|loaded|Warning|warn|deprecat|FutureWarning'

echo "=== [1/6] generate prt planet cache (N=$N) ==="
$PRT "$REPO/src/crossgen_eval/build_eval_cache.py" --engine prt \
    --feature-mode planet --n $N --out "$REPO/data/cache_crossgen_prt_planet" \
    2>&1 | grep -avE "$F" | grep -aE "grid|generated|calibration|wrote"

echo "=== [2/6] generate prt both cache (N=$N) ==="
$PRT "$REPO/src/crossgen_eval/build_eval_cache.py" --engine prt \
    --feature-mode both --n $N --out "$REPO/data/cache_crossgen_prt_both" \
    2>&1 | grep -avE "$F" | grep -aE "grid|generated|calibration|wrote"

echo "=== [3/6] transformer crosstest (prt) ==="
$TORCH "$REPO/src/transformerarch/crosstest.py" --engine prt 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"
echo "=== [4/6] original1dcnn crosstest (prt) ==="
$TORCH "$REPO/src/original1dcnn/crosstest.py" --engine prt 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"
echo "=== [5/6] causal crosstest (prt) ==="
$TORCH "$REPO/src/causal/cnn_trnas/crosstest.py" --engine prt 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"
echo "=== [6/6] 2channel crosstest (prt) ==="
$TORCH "$REPO/src/2channel1dcnn/crosstest.py" --engine prt 2>&1 | grep -E "HEADLINE|Avg|wrote.*details"

echo "=== DONE ==="
