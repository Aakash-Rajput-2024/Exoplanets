"""Cloud-robust retraining of the transformerarch track.

Mirrors src/transformerarch (NasaInaraTransformer, 1-channel `cache_planet`,
MSE, Adam 1e-3 with cosine warmup, grad-clip 1.0, batch 16) on cloud-augmented
spectra. Checkpoints -> src/cloud_robust/transformerarch/checkpoints/.

    python train.py            # full run (50 epochs)  -- slow (batch 16)
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
    train_track("transformerarch", smoke=a.smoke, num_workers=a.num_workers)
