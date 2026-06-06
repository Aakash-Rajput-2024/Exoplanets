import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        print(f"{desc} Starting...")
        for i, item in enumerate(iterable):
            yield item
            if total and (i + 1) % max(1, total // 10) == 0:
                print(f"{desc} Progress: {i+1}/{total} ({(i+1)/total*100:.0f}%)")

from dscm_model import DSCM

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 15
BETA = 1e-4  # KL scaling factor
LATENT_DIM = 64

SUMMARY_PATH = "/Users/aakashrajput/MachineLearning/Exoplanets/data/summary.csv"
SPECTRA_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/data/inara_1by3"
CACHE_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/data/cache_planet"
CHECKPOINT_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/src/10_mil_minus_params/transformer/checkpoints"
# Let's also support the current src/transformerarch/checkpoints path
CHECKPOINT_DIR_ALT = "/Users/aakashrajput/MachineLearning/Exoplanets/src/transformerarch/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_ALT, exist_ok=True)

def load_data_and_envs():
    print("Loading cached spectra and atmosphere targets...")
    train_x = torch.load(os.path.join(CACHE_DIR, "train_x.pt"))
    train_y = torch.load(os.path.join(CACHE_DIR, "train_y.pt"))
    val_x = torch.load(os.path.join(CACHE_DIR, "val_x.pt"))
    val_y = torch.load(os.path.join(CACHE_DIR, "val_y.pt"))
    
    # Load standardized stats to make sure we align targets if needed
    mean_y = torch.load(os.path.join(CACHE_DIR, "mean_y.pt"))
    std_y = torch.load(os.path.join(CACHE_DIR, "std_y.pt"))
    
    mean_x = torch.load(os.path.join(CACHE_DIR, "mean_x.pt"))
    std_x = torch.load(os.path.join(CACHE_DIR, "std_x.pt"))
    
    # Standardize inputs and targets matching train.py
    train_x_std = (train_x - mean_x) / (std_x + 1e-30)
    val_x_std = (val_x - mean_x) / (std_x + 1e-30)
    
    train_y_std = (train_y - mean_y) / (std_y + 1e-8)
    val_y_std = (val_y - mean_y) / (std_y + 1e-8)

    print("Aligning environment variables from summary.csv...")
    summary = pd.read_csv(SUMMARY_PATH, dtype={"planet_index": str})
    summary["planet_index_int"] = summary["planet_index"].astype(int)
    summary = summary.set_index("planet_index_int")
    
    file_names = [f for f in os.listdir(SPECTRA_DIR) if f.endswith('.csv')]
    num_items = len(file_names)
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_items, generator=generator).tolist()
    train_size = int(0.8 * num_items)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Get planet IDs matching train and val splits
    train_planet_ids = [int(file_names[idx].split('.')[0]) for idx in train_indices]
    val_planet_ids = [int(file_names[idx].split('.')[0]) for idx in val_indices]
    
    train_rows = summary.loc[train_planet_ids]
    val_rows = summary.loc[val_planet_ids]
    
    # Columns to standardize
    cont_cols = ["star_temperature", "distance_parsec", "surface_temperature", "surface_pressure"]
    means_e = train_rows[cont_cols].mean()
    stds_e = train_rows[cont_cols].std()
    
    train_cont = (train_rows[cont_cols] - means_e) / (stds_e + 1e-8)
    val_cont = (val_rows[cont_cols] - means_e) / (stds_e + 1e-8)
    
    # One-hot encode star_class
    unique_classes = sorted(summary["star_class"].unique())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    train_class_indices = train_rows["star_class"].map(class_to_idx).values
    val_class_indices = val_rows["star_class"].map(class_to_idx).values
    
    train_class_onehot = np.eye(len(unique_classes))[train_class_indices]
    val_class_onehot = np.eye(len(unique_classes))[val_class_indices]
    
    train_env = np.concatenate([train_cont.values, train_class_onehot], axis=-1)
    val_env = np.concatenate([val_cont.values, val_class_onehot], axis=-1)
    
    train_env_tensor = torch.tensor(train_env, dtype=torch.float32)
    val_env_tensor = torch.tensor(val_env, dtype=torch.float32)
    
    # Save environment normalisation stats for counterfactual generation
    env_stats = {
        "means_e": torch.tensor(means_e.values, dtype=torch.float32),
        "stds_e": torch.tensor(stds_e.values, dtype=torch.float32),
        "class_to_idx": class_to_idx,
        "unique_classes": unique_classes,
        "cont_cols": cont_cols
    }
    torch.save(env_stats, os.path.join(CHECKPOINT_DIR, "dscm_env_stats.pt"))
    torch.save(env_stats, os.path.join(CHECKPOINT_DIR_ALT, "dscm_env_stats.pt"))
    print("Environment normalization stats cached successfully.")
    
    return (TensorDataset(train_x_std, train_y_std, train_env_tensor), 
            TensorDataset(val_x_std, val_y_std, val_env_tensor),
            train_env_tensor.shape[1])

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    train_dataset, val_dataset, e_dim = load_data_and_envs()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    in_channels = train_dataset[0][0].shape[0]
    seq_len = train_dataset[0][0].shape[1]
    a_dim = train_dataset[0][1].shape[0]
    
    print(f"Initializing DSCM: channels={in_channels}, sequence_length={seq_len}, a_dim={a_dim}, e_dim={e_dim}, latent_dim={LATENT_DIM}")
    model = DSCM(sequence_length=seq_len, a_dim=a_dim, e_dim=e_dim, latent_dim=LATENT_DIM).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        for spectra, targets, envs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            spectra, targets, envs = spectra.to(device), targets.to(device), envs.to(device)
            
            optimizer.zero_grad()
            recon_spectra, mu, logvar = model(spectra, targets, envs)
            
            recon_loss = criterion(recon_spectra, spectra)
            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + BETA * kl_loss
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            
        avg_loss = train_loss / len(train_loader)
        avg_recon = train_recon / len(train_loader)
        avg_kl = train_kl / len(train_loader)
        print(f"Epoch {epoch+1} Train | Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | KL: {avg_kl:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for spectra, targets, envs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                spectra, targets, envs = spectra.to(device), targets.to(device), envs.to(device)
                recon_spectra, mu, logvar = model(spectra, targets, envs)
                
                recon_loss = criterion(recon_spectra, spectra)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + BETA * kl_loss
                
                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)
        print(f"Epoch {epoch+1} Val   | Loss: {avg_val_loss:.6f} | Recon: {avg_val_recon:.6f} | KL: {avg_val_kl:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "dscm_best.pth"))
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR_ALT, "dscm_best.pth"))
            print("=> Saved new best DSCM model checkpoint.")
            
    print("DSCM training complete!")

if __name__ == "__main__":
    main()
