"""Cloud-robust retraining of the causal (cnn_trnas) track.

Combines cloud augmentation with the existing env-counterfactual augmentation:
the cloud-augmented planet-mode spectra are CONCATENATED with the pre-built
counterfactual set (src/causal/cnn_trnas/checkpoints/train_x_augmented.pt),
both standardized with the ORIGINAL cache_planet stats, then NasaInaraTransformer
is trained (MSE, cosine warmup, grad-clip 1.0, batch 16).

The DSCM generator and generate_counterfactuals.py are NOT touched -- they are
generators, not the retrieval model. The existing augmented tensors are read-only.
Checkpoints -> src/cloud_robust/causal/checkpoints/.

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
    train_track("causal", smoke=a.smoke, num_workers=a.num_workers)
