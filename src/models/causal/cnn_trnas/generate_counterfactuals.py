import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataloader import load_cached_data_with_envs
from model import DSCM

SUMMARY_PATH = "/Users/aakashrajput/MachineLearning/Exoplanets/data/summary.csv"
CACHE_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/data/cache_planet"
CHECKPOINT_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/src/models/causal/cnn_trnas/checkpoints"
LATENT_DIM = 64
MAHALANOBIS_THRESHOLD = 3.0  # drop counterfactuals whose (A, E') pair is implausible


def fit_joint_gaussian(joint, ridge=1e-4):
    mean = joint.mean(dim=0, keepdim=True)
    centered = joint - mean
    n = joint.size(0)
    cov = (centered.t() @ centered) / max(1, n - 1)
    cov = cov + torch.eye(cov.size(0), device=cov.device) * ridge
    cov_inv = torch.linalg.inv(cov)
    return mean, cov_inv


def mahalanobis(joint, mean, cov_inv):
    diff = joint - mean
    return torch.sqrt(torch.clamp((diff @ cov_inv * diff).sum(dim=1), min=0.0))

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # 1. Load original data and environment variables
    train_dataset, val_dataset, env_stats = load_cached_data_with_envs(CACHE_DIR, SUMMARY_PATH)
    
    train_x = train_dataset.tensors[0]
    train_y = train_dataset.tensors[1]
    train_env = train_dataset.tensors[2]
    
    seq_len = train_x.shape[2]
    a_dim = train_y.shape[1]
    e_dim = train_env.shape[1]
    
    # 2. Load trained DSCM model
    model = DSCM(sequence_length=seq_len, a_dim=a_dim, e_dim=e_dim, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "dscm_best.pth"), map_location=device))
    model.eval()
    
    # 3. Create counterfactual environments E' by shuffling E. Then filter
    #    implausible (A, E') pairs using the empirical joint Gaussian of (A, E).
    num_samples = train_x.shape[0]
    shuffled_idx = torch.randperm(num_samples)
    train_env_cf = train_env[shuffled_idx]

    joint_full = torch.cat([train_y, train_env], dim=1)
    joint_mean, joint_cov_inv = fit_joint_gaussian(joint_full)
    joint_cf = torch.cat([train_y, train_env_cf], dim=1)
    mahal_cf = mahalanobis(joint_cf, joint_mean, joint_cov_inv)
    keep_mask = mahal_cf < MAHALANOBIS_THRESHOLD
    n_keep = int(keep_mask.sum().item())
    print(f"Plausibility filter: kept {n_keep}/{num_samples} counterfactuals "
          f"(Mahalanobis < {MAHALANOBIS_THRESHOLD}).")

    print("Generating counterfactual spectra...")
    cf_spectra_list = []
    cf_y_list = []

    # Process in batches to avoid OOM
    batch_size = 128
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_mask = keep_mask[i:end_idx]
            if not batch_mask.any():
                continue

            s_batch = train_x[i:end_idx][batch_mask].to(device)
            a_batch = train_y[i:end_idx][batch_mask].to(device)
            e_batch = train_env[i:end_idx][batch_mask].to(device)
            e_cf_batch = train_env_cf[i:end_idx][batch_mask].to(device)

            # Abduction: sample exogenous noise from the posterior q(z | s, a, e).
            # Using a sample (not the MAP mu) keeps the SCM stochastic.
            mu, logvar = model.encoder(s_batch, a_batch, e_batch)
            z = model.reparameterize(mu, logvar)

            # Prediction under factual and counterfactual environments
            recon_s_original = model.decoder(z, a_batch, e_batch)
            recon_s_cf = model.decoder(z, a_batch, e_cf_batch)

            # Additive-noise SCM: S = g(Z, A, E) + eta. The factual residual eta
            # is carried over to the counterfactual: S_cf = S + g(Z,A,E') - g(Z,A,E).
            s_cf = s_batch + (recon_s_cf - recon_s_original)
            cf_spectra_list.append(s_cf.cpu())
            cf_y_list.append(a_batch.cpu())

            if (i + batch_size) % 10240 == 0 or end_idx == num_samples:
                print(f"Processed {end_idx}/{num_samples} samples...")

    cf_spectra = torch.cat(cf_spectra_list, dim=0)
    cf_y = torch.cat(cf_y_list, dim=0)

    # 4. Augment the training set: concatenate originals + filtered counterfactuals.
    train_x_augmented = torch.cat([train_x, cf_spectra], dim=0)
    train_y_augmented = torch.cat([train_y, cf_y], dim=0)

    print(f"Original train size: {train_x.shape[0]}")
    print(f"Counterfactuals kept: {cf_spectra.shape[0]}")
    print(f"Augmented train size: {train_x_augmented.shape[0]}")
    
    # Save the augmented dataset
    torch.save(train_x_augmented, os.path.join(CHECKPOINT_DIR, "train_x_augmented.pt"))
    torch.save(train_y_augmented, os.path.join(CHECKPOINT_DIR, "train_y_augmented.pt"))
    print("Augmented dataset saved successfully!")

if __name__ == "__main__":
    main()
