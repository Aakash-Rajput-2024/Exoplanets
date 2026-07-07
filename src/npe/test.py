import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sbi.analysis import pairplot

def main():
    print("Testing LC-NPE Model on Real Validation Data...")
    
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "lc_npe_posterior.pt")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Posterior not found at {CHECKPOINT_PATH}. Please run train_lcnpe.py first.")
        return
        
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from npe.model import TransformerSummaryNetwork, build_neural_posterior
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    posterior = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print("Loaded LC-NPE Posterior successfully!")
    
    # Load Validation Data
    val_x_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache_planet/val_x.pt"))
    val_y_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache_planet/val_y.pt"))
    
    if not os.path.exists(val_x_path):
        print("Validation data not found. Please ensure the cache is built.")
        return
        
    print("Loading Validation Dataset and Normalization Stats...")
    val_x_raw = torch.load(val_x_path, map_location=device, weights_only=False)
    val_y_raw = torch.load(val_y_path, map_location=device, weights_only=False)
    
    mean_x = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache_planet/mean_x.pt")), map_location=device, weights_only=False)
    std_x = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache_planet/std_x.pt")), map_location=device, weights_only=False)
    mean_y = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache_planet/mean_y.pt")), map_location=device, weights_only=False)
    std_y = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache_planet/std_y.pt")), map_location=device, weights_only=False)
    
    # The NPE model was trained on Standardized data (Mean 0, Std 1). 
    # We MUST standardize the validation spectra before feeding them to the model!
    val_x_std = (val_x_raw - mean_x) / (std_x + 1e-30)
    
    # Evaluate on a subset of validation data to save time (e.g. 100 samples)
    NUM_EVAL = 100
    print(f"Evaluating R^2 and RMSE on {NUM_EVAL} validation samples...")
    
    val_x_sub = val_x_std[:NUM_EVAL]
    val_y_sub = val_y_raw[:NUM_EVAL]  # Keep True Y in physical scale for metrics!
    
    predictions = []
    
    for i in range(NUM_EVAL):
        # We draw 500 samples from the standardized probability distribution
        samples_std = posterior.sample((500,), x=val_x_sub[i].unsqueeze(0), show_progress_bars=False)
        
        # The 'prediction' is the mean of the standardized posterior
        pred_y_std = samples_std.mean(dim=0)
        
        # INVERSE TRANSFORM: Convert standardized prediction back to physical scale!
        pred_y_physical = pred_y_std * std_y.squeeze(0) + mean_y.squeeze(0)
        
        predictions.append(pred_y_physical.cpu().numpy())
        
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{NUM_EVAL} spectra...")
            
    predictions = np.array(predictions)
    truths = val_y_sub.cpu().numpy()
    
    target_columns = [
        'H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 
        'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2'
    ]
    
    # Calculate Metrics per molecule
    r2 = r2_score(truths, predictions, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(truths, predictions, multioutput='raw_values'))
    mae = mean_absolute_error(truths, predictions, multioutput='raw_values')
    
    print("\n" + "="*60)
    print("LC-NPE VALIDATION METRICS (Per Molecule)")
    print("="*60)
    print(f"{'Molecule':<10} | {'R2 Score':<12} | {'RMSE':<10} | {'MAE':<10}")
    print("-" * 60)
    for i, mol in enumerate(target_columns):
        print(f"{mol:<10} | {r2[i]:<12.4f} | {rmse[i]:<10.4f} | {mae[i]:<10.4f}")
    print("="*60)
    
    # Generate Corner Plot for the first validation sample
    print("\nGenerating Corner Plot for Sample 1...")
    samples_plot = posterior.sample((2000,), x=val_x_sub[0].unsqueeze(0), show_progress_bars=False)
    
    # Get true values for the first sample to overlay on the plot
    true_values = truths[0]
    
    fig, axes = pairplot(
        samples_plot.cpu().numpy(), 
        limits=[[-3, 3]]*12, 
        figsize=(15, 15), 
        title="LC-NPE Posterior Corner Plot (vs True Values)"
    )
    
    save_path = os.path.join(os.path.dirname(__file__), "posterior_corner_plot.png")
    fig.savefig(save_path)
    print(f"Saved corner plot to {save_path}")

if __name__ == "__main__":
    main()
