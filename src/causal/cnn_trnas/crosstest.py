"""Cross-generator evaluation for the causal (augmented transformer) model.

The predictive model's forward(x) takes only the spectrum (env features are DSCM
*training* machinery and are ignored at eval time, per test.py), so no environment
synthesis is needed here.

  python src/causal/cnn_trnas/crosstest.py --engine taurex

Run in the same PyTorch env as test.py (e.g. venvNLP).
"""
import os
import sys
import argparse

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
sys.path.insert(0, os.path.join(REPO, "src", "crossgen_eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crosseval_common import run_crosstest
from model import NasaInaraTransformer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["taurex", "prt"], default="taurex")
    args = ap.parse_args()

    cache_dir = os.path.join(REPO, f"data/cache_crossgen_{args.engine}_planet")
    model = NasaInaraTransformer(in_channels=1, sequence_length=4379)
    run_crosstest(
        model=model,
        checkpoint_path=os.path.join(HERE, "checkpoints", "model_best.pth"),
        cache_dir=cache_dir,
        inara_stats_dir=os.path.join(REPO, "data", "cache_planet"),
        out_dir=HERE,
        model_name=f"NasaInaraTransformer (causal, 1-ch planet) [eval={args.engine}]",
    )


if __name__ == "__main__":
    main()
