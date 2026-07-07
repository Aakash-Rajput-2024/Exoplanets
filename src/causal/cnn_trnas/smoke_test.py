"""
End-to-end smoke test: DSCM → CFs → Transformer, ~500 samples, 3 epochs each.
Run:  python smoke_test.py
"""
import os, sys, math, textwrap
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

SUMMARY_PATH = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/summary.csv"
CACHE_DIR    = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/cache_planet"

N_TRAIN      = 500
N_VAL        = 200
DSCM_EPOCHS  = 3
TRANS_EPOCHS = 3
BATCH        = 32
LATENT_DIM   = 64
BETA_MAX     = 1.0
LAMBDA_IND   = 1.0
MAHAL_THRESH = 3.0

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import load_cached_data_with_envs
from model import DSCM, NasaInaraTransformer

# ─── helpers ────────────────────────────────────────────────────────────────

def _pairwise_sq_dist(x):
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^Tb  (MPS-compatible, avoids cdist)
    sq = (x * x).sum(dim=1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * (x @ x.t())
    return d2.clamp(min=0)

def hsic(z, e, eps=1e-8):
    n = z.size(0)
    if n < 4:
        return torch.zeros((), device=z.device)
    def _rbf(x):
        d2 = _pairwise_sq_dist(x)
        s2 = d2.detach().flatten().median().clamp_min(eps)
        return torch.exp(-d2 / (2.0 * s2 + eps))
    K, L = _rbf(z), _rbf(e)
    H = torch.eye(n, device=z.device) - 1.0 / n
    return (K @ H * (L @ H).t()).sum() / max(1, (n - 1) ** 2)

def fit_gaussian(joint, ridge=1e-4):
    mean = joint.mean(0, keepdim=True)
    c = joint - mean
    cov = (c.t() @ c) / max(1, len(joint) - 1)
    cov_inv = torch.linalg.inv(cov + torch.eye(cov.size(0)) * ridge)
    return mean, cov_inv

def mahalanobis(joint, mean, cov_inv):
    d = joint - mean
    return torch.sqrt((d @ cov_inv * d).sum(1).clamp(min=0))

def sep(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)

# ─── load data ───────────────────────────────────────────────────────────────

sep("Loading data")
train_ds, val_ds, env_stats = load_cached_data_with_envs(CACHE_DIR, SUMMARY_PATH)

rng = torch.Generator().manual_seed(42)
train_sub = Subset(train_ds, torch.randperm(len(train_ds), generator=rng)[:N_TRAIN].tolist())
val_sub   = Subset(val_ds,   torch.randperm(len(val_ds),   generator=rng)[:N_VAL].tolist())

def subset_tensors(sub):
    xs, ys, es = [], [], []
    for x, y, e in sub:
        xs.append(x); ys.append(y); es.append(e)
    return torch.stack(xs), torch.stack(ys), torch.stack(es)

train_x, train_y, train_e = subset_tensors(train_sub)
val_x,   val_y,   val_e   = subset_tensors(val_sub)

seq_len = train_x.shape[2]
a_dim   = train_y.shape[1]
e_dim   = train_e.shape[1]
in_ch   = train_x.shape[1]
print(f"Train: {N_TRAIN} samples | Val: {N_VAL} samples")
print(f"S shape: {list(train_x.shape[1:])}  A dim: {a_dim}  E dim: {e_dim}")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

# ─── Phase 1: train DSCM ────────────────────────────────────────────────────

sep("Phase 1 — Train DSCM")
dscm = DSCM(sequence_length=seq_len, a_dim=a_dim, e_dim=e_dim, latent_dim=LATENT_DIM).to(device)
opt  = optim.Adam(dscm.parameters(), lr=5e-4)
mse  = nn.MSELoss()
train_loader = DataLoader(TensorDataset(train_x, train_y, train_e),
                          batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_x, val_y, val_e),
                          batch_size=BATCH, shuffle=False)

for ep in range(1, DSCM_EPOCHS + 1):
    beta = BETA_MAX * min(1.0, ep / max(1, 3))
    dscm.train()
    t_recon = t_kl = t_hsic = t_n = 0
    for s, a, e in train_loader:
        s, a, e = s.to(device), a.to(device), e.to(device)
        opt.zero_grad()
        mu, logvar = dscm.encoder(s, a, e)
        z  = dscm.reparameterize(mu, logvar)
        rs = dscm.decoder(z, a, e)
        rl = mse(rs, s)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        il = hsic(z, e)
        loss = rl + beta * kl + LAMBDA_IND * il
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dscm.parameters(), 1.0)
        opt.step()
        t_recon += rl.item(); t_kl += kl.item(); t_hsic += il.item(); t_n += 1

    dscm.eval()
    v_recon = v_kl = v_hsic = v_n = 0
    with torch.no_grad():
        for s, a, e in val_loader:
            s, a, e = s.to(device), a.to(device), e.to(device)
            mu, logvar = dscm.encoder(s, a, e)
            z  = dscm.reparameterize(mu, logvar)
            rs = dscm.decoder(z, a, e)
            rl = mse(rs, s)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            il = hsic(z, e)
            v_recon += rl.item(); v_kl += kl.item(); v_hsic += il.item(); v_n += 1

    print(
        f"Ep {ep}/{DSCM_EPOCHS} | "
        f"Train recon={t_recon/t_n:.4f}  KL={t_kl/t_n:.4f}  HSIC={t_hsic/t_n:.6f}  β={beta:.2f} | "
        f"Val  recon={v_recon/v_n:.4f}  KL={v_kl/v_n:.4f}  HSIC={v_hsic/v_n:.6f}"
    )

# ─── Phase 2: generate counterfactuals ──────────────────────────────────────

sep("Phase 2 — Generate Counterfactuals")
dscm.eval()

shuffled_idx  = torch.randperm(N_TRAIN)
train_e_cf    = train_e[shuffled_idx]

joint_real = torch.cat([train_y, train_e], dim=1)
jmean, jcov_inv = fit_gaussian(joint_real)
joint_cf    = torch.cat([train_y, train_e_cf], dim=1)
mahal_cf    = mahalanobis(joint_cf, jmean, jcov_inv)
keep        = mahal_cf < MAHAL_THRESH
print(f"OOD filter: kept {keep.sum().item()}/{N_TRAIN} CF pairs (Mahal < {MAHAL_THRESH})")

cf_s_list, cf_y_list = [], []
with torch.no_grad():
    for i in range(0, N_TRAIN, BATCH):
        bm = keep[i:i+BATCH]
        if not bm.any():
            continue
        s = train_x[i:i+BATCH][bm].to(device)
        a = train_y[i:i+BATCH][bm].to(device)
        e = train_e[i:i+BATCH][bm].to(device)
        ecf = train_e_cf[i:i+BATCH][bm].to(device)
        mu, logvar = dscm.encoder(s, a, e)
        z = dscm.reparameterize(mu, logvar)
        delta = dscm.decoder(z, a, ecf) - dscm.decoder(z, a, e)
        cf_s_list.append((s + delta).cpu())
        cf_y_list.append(a.cpu())

cf_s = torch.cat(cf_s_list)
cf_y = torch.cat(cf_y_list)

# Diagnostics: how much do CFs differ from originals?
orig_kept = train_x[keep]
diff = (cf_s - orig_kept).abs()
print(f"CF delta stats — mean: {diff.mean():.4f}  max: {diff.max():.4f}  "
      f"95th pct: {diff.flatten().quantile(0.95):.4f}")
print(f"  (near-zero delta = DSCM not yet well-trained; expected at 3 epochs)")

aug_x = torch.cat([train_x, cf_s])
aug_y = torch.cat([train_y, cf_y])
print(f"Augmented dataset: {len(aug_x)} samples ({N_TRAIN} orig + {len(cf_s)} CF)")

# ─── Phase 3: train Transformer ─────────────────────────────────────────────

sep("Phase 3 — Train NasaInaraTransformer")
model = NasaInaraTransformer(in_channels=in_ch, sequence_length=seq_len).to(device)
opt2  = optim.Adam(model.parameters(), lr=1e-3)

aug_loader = DataLoader(TensorDataset(aug_x, aug_y),
                        batch_size=BATCH, shuffle=True)
val_loader2 = DataLoader(TensorDataset(val_x, val_y),
                         batch_size=BATCH, shuffle=False)

for ep in range(1, TRANS_EPOCHS + 1):
    model.train()
    t_loss = t_n = 0
    for x, y in aug_loader:
        x, y = x.to(device), y.to(device)
        opt2.zero_grad()
        loss = mse(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        t_loss += loss.item(); t_n += 1

    model.eval()
    v_loss = v_n = 0
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for x, y in val_loader2:
            x, y = x.to(device), y.to(device)
            p = model(x)
            v_loss += mse(p, y).item(); v_n += 1
            preds_all.append(p.cpu()); tgts_all.append(y.cpu())

    print(f"Ep {ep}/{TRANS_EPOCHS} | Train MSE: {t_loss/t_n:.6f}  Val MSE: {v_loss/v_n:.6f}")

# ─── Final diagnostics ───────────────────────────────────────────────────────

sep("Final Diagnostics")
preds = torch.cat(preds_all)
tgts  = torch.cat(tgts_all)

gases = ['H2O','CO2','O2','N2','CH4','N2O','CO','O3','SO2','NH3','C2H6','NO2']
print(f"\n{'Gas':<8} {'Val MAE':>10} {'Val RMSE':>10}")
print("─" * 32)
for i, gas in enumerate(gases):
    mae  = (preds[:, i] - tgts[:, i]).abs().mean().item()
    rmse = ((preds[:, i] - tgts[:, i]).pow(2).mean()).sqrt().item()
    print(f"{gas:<8} {mae:>10.4f} {rmse:>10.4f}")

overall_mae  = (preds - tgts).abs().mean().item()
overall_rmse = ((preds - tgts).pow(2).mean()).sqrt().item()
print(f"\n{'Overall':<8} {overall_mae:>10.4f} {overall_rmse:>10.4f}")

# Sanity: are predictions non-trivial (not all the same)?
pred_std = preds.std(dim=0).mean().item()
tgt_std  = tgts.std(dim=0).mean().item()
print(f"\nPred std (mean over gases): {pred_std:.4f}  "
      f"(target std: {tgt_std:.4f})")
if pred_std < 0.01 * tgt_std:
    print("  [WARN] Predictions have near-zero variance — model may be predicting the mean.")
else:
    print("  [OK] Predictions show reasonable spread.")

print("\n✓ Smoke test complete.")
