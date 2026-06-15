import os
import torch
from sbi.analysis import pairplot

def main():
    print("Testing LC-NPE Model...")
    
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "lc_npe_posterior.pt")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Posterior not found at {CHECKPOINT_PATH}. Please run train_lcnpe.py first.")
        return
        
    posterior = torch.load(CHECKPOINT_PATH, map_location="cpu")
    print("Loaded LC-NPE Posterior successfully!")
    
    # Create a dummy test spectrum to evaluate the posterior
    # In a real scenario, this would be an actual validation sample
    print("Generating a sample posterior distribution...")
    dummy_spectrum = torch.randn(1, 1, 4379)
    
    # Sample from the posterior
    # This generates 1000 potential atmospheric compositions that match the spectrum
    samples = posterior.sample((1000,), x=dummy_spectrum)
    
    # Plot the corner plot using sbi's built-in pairplot
    fig, axes = pairplot(samples, limits=[[-3, 3]]*12, figsize=(15, 15), title="LC-NPE Posterior Corner Plot")
    
    save_path = os.path.join(os.path.dirname(__file__), "posterior_corner_plot.png")
    fig.savefig(save_path)
    print(f"Saved corner plot to {save_path}")

if __name__ == "__main__":
    main()
