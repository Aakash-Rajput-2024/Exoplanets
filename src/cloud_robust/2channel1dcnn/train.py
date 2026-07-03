"""Cloud-robust retraining of the 2channel1dcnn track.

Mirrors src/2channel1dcnn (NasaInaraModel, 2-channel `cache` = [star_planet,
planet], L1, Adam 1e-3, batch 1024) on cloud-augmented spectra. Only the
planet_signal channel is clouded; the star-dominated channel is unchanged.
Checkpoints -> src/cloud_robust/2channel1dcnn/checkpoints/.

    python train.py            # full run (50 epochs)
    python train.py --smoke    # 1-epoch subset sanity check
"""
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_common import train_track

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a tiny subset")
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()
    train_track("2channel1dcnn", smoke=a.smoke, num_workers=a.num_workers)
