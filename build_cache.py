import os
import sys

# Import the cache_data function from one of the dataloaders
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "transformerarch"))
from dataloader import cache_data

def main():
    print("Building cached tensors from raw CSV files...")
    
    # Set the paths based on what you have
    sum_path = os.path.join("data", "summary.csv")
    dir_path = os.path.join("data", "Total_dataset")
    save_dir = os.path.join("data", "cache_planet")
    
    if not os.path.exists(dir_path):
        print(f"ERROR: Could not find the raw CSV folder at {dir_path}")
        print("Please ensure your CSV files are inside the 'data/Total_dataset' folder!")
        return
        
    if not os.path.exists(sum_path):
        print(f"ERROR: Could not find summary file at {sum_path}")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # We will use feature_mode "planet" since the causal model uses 1 channel usually, 
    # but the partner's model uses 1 channel. Wait, let's use "planet" feature mode.
    # The transformer and causal model are 1-channel models.
    print(f"Reading from {dir_path}")
    print(f"Saving cache to {save_dir}")
    
    # Let's call the function
    cache_data(sum_path, dir_path, save_dir, feature_mode="planet")
    
    print("\nSUCCESS! The data has been successfully cached and is ready for the LC-NPE model.")

if __name__ == "__main__":
    main()
