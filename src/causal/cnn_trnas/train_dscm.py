import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

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

from dataloader import load_cached_data_with_envs
from model import DSCM

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 15
BETA_MAX = 1.0          # target KL weight (standard ELBO uses 1.0)
BETA_WARMUP_EPOCHS = 5  # linear ramp 0 -> BETA_MAX
LAMBDA_IND = 1.0        # weight on HSIC(Z, E) independence penalty
GRAD_CLIP_MAX = 1.0
LATENT_DIM = 64

SUMMARY_PATH = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/summary.csv"
CACHE_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/cache_planet"
CHECKPOINT_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/causal/cnn_trnas/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _pairwise_sq_dist(x):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^Tb  (avoids cdist, MPS-compatible)
    sq = (x * x).sum(dim=1, keepdim=True)
    return (sq + sq.t() - 2.0 * (x @ x.t())).clamp(min=0)


def hsic(z, e, eps=1e-8):
    # Empirical HSIC with RBF kernels and the median heuristic for bandwidths.
    # Penalizes statistical dependence (linear and nonlinear) between Z and E.
    n = z.size(0)
    if n < 4:
        return torch.zeros((), device=z.device)

    def _rbf(x):
        d2 = _pairwise_sq_dist(x)
        sigma2 = d2.detach().flatten().median().clamp_min(eps)
        return torch.exp(-d2 / (2.0 * sigma2 + eps))

    K = _rbf(z)
    L = _rbf(e)
    H = torch.eye(n, device=z.device) - 1.0 / n
    KH = K @ H
    LH = L @ H
    return (KH * LH.t()).sum() / max(1, (n - 1) ** 2)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    train_dataset, val_dataset, env_stats = load_cached_data_with_envs(CACHE_DIR, SUMMARY_PATH)
    
    # Save environment stats for counterfactuals
    torch.save(env_stats, os.path.join(CHECKPOINT_DIR, "dscm_env_stats.pt"))
    print("Saved environment normalization stats.")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    in_channels = train_dataset[0][0].shape[0]
    seq_len = train_dataset[0][0].shape[1]
    a_dim = train_dataset[0][1].shape[0]
    e_dim = train_dataset[0][2].shape[0]
    
    print(f"Initializing DSCM: channels={in_channels}, sequence_length={seq_len}, a_dim={a_dim}, e_dim={e_dim}, latent_dim={LATENT_DIM}")
    model = DSCM(sequence_length=seq_len, a_dim=a_dim, e_dim=e_dim, latent_dim=LATENT_DIM).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # KL warmup: 0 -> BETA_MAX linearly across BETA_WARMUP_EPOCHS
        beta = BETA_MAX * min(1.0, (epoch + 1) / max(1, BETA_WARMUP_EPOCHS))

        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_ind = 0.0

        for spectra, targets, envs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            spectra, targets, envs = spectra.to(device), targets.to(device), envs.to(device)

            optimizer.zero_grad()
            mu, logvar = model.encoder(spectra, targets, envs)
            z = model.reparameterize(mu, logvar)
            recon_spectra = model.decoder(z, targets, envs)

            recon_loss = criterion(recon_spectra, spectra)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # Z must be independent of E for E-interventions to be valid SCM operations
            ind_loss = hsic(z, envs)

            loss = recon_loss + beta * kl_loss + LAMBDA_IND * ind_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX)
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            train_ind += ind_loss.item()

        avg_loss = train_loss / len(train_loader)
        avg_recon = train_recon / len(train_loader)
        avg_kl = train_kl / len(train_loader)
        avg_ind = train_ind / len(train_loader)
        print(f"Epoch {epoch+1} Train | Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | KL: {avg_kl:.4f} | HSIC(Z,E): {avg_ind:.6f} | beta: {beta:.3f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_ind = 0.0

        with torch.no_grad():
            for spectra, targets, envs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                spectra, targets, envs = spectra.to(device), targets.to(device), envs.to(device)
                mu, logvar = model.encoder(spectra, targets, envs)
                z = model.reparameterize(mu, logvar)
                recon_spectra = model.decoder(z, targets, envs)

                recon_loss = criterion(recon_spectra, spectra)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                ind_loss = hsic(z, envs)
                loss = recon_loss + beta * kl_loss + LAMBDA_IND * ind_loss

                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
                val_ind += ind_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)
        avg_val_ind = val_ind / len(val_loader)
        print(f"Epoch {epoch+1} Val   | Loss: {avg_val_loss:.6f} | Recon: {avg_val_recon:.6f} | KL: {avg_val_kl:.4f} | HSIC(Z,E): {avg_val_ind:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "dscm_best.pth"))
            print("=> Saved new best DSCM model checkpoint.")
            
    print("DSCM training complete!")

if __name__ == "__main__":
    main()
