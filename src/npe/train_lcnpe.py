import os
import torch
from sbi.inference import SNPE
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from npe.model import TransformerSummaryNetwork, build_neural_posterior

# We expect the causal counterfactual script to generate these augmented datasets
AUGMENTED_X_PATH = "../causal/cnn_trnas/checkpoints/train_x_augmented.pt"
AUGMENTED_Y_PATH = "../causal/cnn_trnas/checkpoints/train_y_augmented.pt"
CHECKPOINT_DIR = os.path.dirname(__file__)

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Loading LC-NPE Training script...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 1. Load the Augmented Data (produced by the Causal DSCM)
    aug_x_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), AUGMENTED_X_PATH))
    aug_y_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), AUGMENTED_Y_PATH))
    
    if not os.path.exists(aug_x_full_path):
        print(f"Augmented data not found at {aug_x_full_path}!")
        print("For demonstration, generating dummy data. Run the counterfactual generator later for real data.")
        x_train = torch.randn(1000, 1, 4379)
        y_train = torch.randn(1000, 12)
    else:
        print("Loading real causally augmented data...")
        # Load with weights_only=True to avoid future warnings
        x_train = torch.load(aug_x_full_path, map_location="cpu", weights_only=False)
        y_train = torch.load(aug_y_full_path, map_location="cpu", weights_only=False)
        
    # [MEMORY FIX] If the dataset is too massive for a 6GB GPU, safely subsample it to 20,000 samples.
    # The augmented data has 61,000+ samples, which throws an OOM when calculating Z-scores in SBI.
    MAX_SAMPLES = 20000
    if x_train.shape[0] > MAX_SAMPLES:
        print(f"Dataset is very large ({x_train.shape[0]} samples). Subsampling to {MAX_SAMPLES} to prevent CUDA Out Of Memory.")
        # Shuffle before subsampling to ensure good coverage
        indices = torch.randperm(x_train.shape[0])[:MAX_SAMPLES]
        x_train = x_train[indices]
        y_train = y_train[indices]
        
    print(f"Data shape: X={x_train.shape}, Y={y_train.shape}")
    
    # 2. Build the Transformer Summary Network
    in_channels = x_train.shape[1] if x_train.dim() == 3 else 1
    seq_len = x_train.shape[-1]
    
    print("Building Transformer Summary Network...")
    summary_net = TransformerSummaryNetwork(in_channels=in_channels, sequence_length=seq_len)
    
    # 3. Build the Neural Posterior (Neural Spline Flow)
    print("Initializing Neural Spline Flow (NSF) Posterior...")
    density_estimator_build_fn = build_neural_posterior(summary_net, model_type='nsf')
    
    inference = SNPE(density_estimator=density_estimator_build_fn, device=device)
    
    # 4. Feed the training data to SBI
    print("Appending data to SNPE...")
    inference = inference.append_simulations(y_train, x_train)
    
    # 5. Train the Normalizing Flow
    print("Training the Normalizing Flow. This may take a while...")
    density_estimator = inference.train(training_batch_size=64, learning_rate=5e-4)
    
    # 6. Build and Save the Posterior
    posterior = inference.build_posterior(density_estimator)
    
    save_path = os.path.join(CHECKPOINT_DIR, "lc_npe_posterior.pt")
    torch.save(posterior, save_path)
    print(f"LC-NPE Training Complete! Posterior saved to {save_path}")

if __name__ == "__main__":
    main()
