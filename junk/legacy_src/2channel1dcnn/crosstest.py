"""Cross-generator evaluation for the 2-channel 1D-CNN model.

Uses the 'both' cross-gen cache (channels = [star_planet_signal, planet_signal]),
so the eval set must be generated with --feature-mode both.

  python src/2channel1dcnn/crosstest.py --engine taurex

Run in the same PyTorch env as test.py (e.g. venvNLP).
"""
import os
import sys
import argparse

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
sys.path.insert(0, os.path.join(REPO, "src", "crossgen_eval"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crosseval_common import run_crosstest
from model import NasaInaraModel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["taurex", "prt"], default="taurex")
    args = ap.parse_args()

    cache_dir = os.path.join(REPO, f"data/cache_crossgen_{args.engine}_both")
    model = NasaInaraModel(in_channels=2, sequence_length=4379)
    run_crosstest(
        model=model,
        checkpoint_path=os.path.join(HERE, "checkpoints", "model_best.pth"),
        cache_dir=cache_dir,
        inara_stats_dir=os.path.join(REPO, "data", "cache"),
        out_dir=HERE,
        model_name=f"NasaInaraModel (2-ch both) [eval={args.engine}]",
    )


if __name__ == "__main__":
    main()
