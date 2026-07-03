"""Cloud-robust retraining of the original1dcnn track.

Mirrors src/original1dcnn (NasaInaraModel, L1, Adam 1e-3, batch 1024) but trains
on cloud-augmented spectra. Checkpoints -> src/cloud_robust/original1dcnn/checkpoints/.
The original track is read-only.

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
    train_track("original1dcnn", smoke=a.smoke, num_workers=a.num_workers)
