import os
import torch

def main():
    save_dir = os.path.join("data", "cache_planet")
    
    print("Loading cached tensors to compute statistics...")
    train_x_path = os.path.join(save_dir, "train_x.pt")
    train_y_path = os.path.join(save_dir, "train_y.pt")
    
    if not os.path.exists(train_x_path):
        print(f"Error: {train_x_path} not found.")
        return
        
    train_x = torch.load(train_x_path)
    train_y = torch.load(train_y_path)
    
    print("Computing mean and std for inputs (channel-wise)...")
    mean_x = train_x.mean(dim=(0, 2), keepdim=True)
    std_x = train_x.std(dim=(0, 2), keepdim=True)
    
    print("Computing mean and std for targets (column-wise)...")
    mean_y = train_y.mean(dim=0, keepdim=True)
    std_y = train_y.std(dim=0, keepdim=True)
    
    print("Saving statistics...")
    torch.save(mean_x, os.path.join(save_dir, "mean_x.pt"))
    torch.save(std_x, os.path.join(save_dir, "std_x.pt"))
    torch.save(mean_y, os.path.join(save_dir, "mean_y.pt"))
    torch.save(std_y, os.path.join(save_dir, "std_y.pt"))
    
    print("SUCCESS! The statistics have been successfully computed and saved.")

if __name__ == "__main__":
    main()
