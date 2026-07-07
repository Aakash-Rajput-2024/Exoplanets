import torchinfo
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_cached_data
from model import NasaInaraModel

SUMMARY_PATH = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/summary.csv"
SPECTRA_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/inara_1by3"
CACHE_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/cache"
CHARTS_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/2channel1dcnn/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

NORMALIZE_INPUTS = True

TARGET_COLUMNS = [
    'H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 
    'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2'
]

def calculate_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return r2, rmse

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_dataset = load_cached_data(
        CACHE_DIR, SUMMARY_PATH, SPECTRA_DIR, 
        normalize_inputs=NORMALIZE_INPUTS
    )
    
    in_channels = val_dataset[0][0].shape[0]
    seq_len = val_dataset[0][0].shape[1]
    print(f"Detected channels: {in_channels}, sequence length: {seq_len}")
    model = NasaInaraModel(in_channels=in_channels, sequence_length=seq_len).to(device)
    # Generate torchinfo summary string
    try:
        model_summary_str = str(torchinfo.summary(model, input_size=(1, in_channels, seq_len), device="cpu", verbose=0))
    except Exception as e:
        model_summary_str = f"Error generating model summary: {e}"
    
    
    checkpoint_path = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/2channel1dcnn/checkpoints/model_best.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded model from checkpoint.")
    else:
        print("No checkpoint found. Evaluating random model.")

    eval_size = min(5000, len(val_dataset))
    eval_subset, _ = random_split(
        val_dataset, [eval_size, len(val_dataset) - eval_size], generator=torch.Generator().manual_seed(42)
    )
    eval_loader = DataLoader(eval_subset, batch_size=64, shuffle=False)

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for spectra, targets in eval_loader:
            spectra, targets = spectra.to(device), targets.to(device)
            preds = model(spectra)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    
    # Load scaling stats and inverse-transform targets back to linear scale
    mean_y = torch.load(os.path.join(CACHE_DIR, "mean_y.pt")).cpu().numpy()
    std_y = torch.load(os.path.join(CACHE_DIR, "std_y.pt")).cpu().numpy()
    y_pred = y_pred * std_y + mean_y
    y_true = y_true * std_y + mean_y
    
    r2_scores, rmse_scores = calculate_metrics(y_true, y_pred)
    mae_scores = np.mean(np.abs(y_true - y_pred), axis=0)
    label_suffix = " (abundance)"

    # Generate details.txt standardized performance report
    report_content = f"""================================================================================
EXOPLANET ATMOSPHERIC RETRIEVAL EVALUATION REPORT
================================================================================
Model: NasaInaraModel (2-Channel, Wavelength <= 2.0 um)
Features: 2 channels (star_planet_signal + planet_signal)
Sequence Length: {seq_len}
Channels: {in_channels}
Targets: Standardized Column-wise Linear Abundances (12 Species)
Evaluation Subset Size: {eval_size} samples (validation split)
--------------------------------------------------------------------------------

OVERALL PERFORMANCE METRICS:
--------------------------------------------------------------------------------
Average R2 Score:   {np.mean(r2_scores):.4f}
Average RMSE:       {np.mean(rmse_scores):.4f}
Average MAE (L1):   {np.mean(mae_scores):.4f}
--------------------------------------------------------------------------------

INDIVIDUAL SPECIES PERFORMANCE:
--------------------------------------------------------------------------------
{"Species":<11} | {"R2 Score":<8} | {"RMSE":<8} | {"MAE (L1)":<10}
--------------------------------------------------------------------------------
"""
    for i, col in enumerate(TARGET_COLUMNS):
        report_content += f"{col:<11} | {r2_scores[i]:<8.4f} | {rmse_scores[i]:<8.4f} | {mae_scores[i]:<10.4f}\n"
    report_content += "================================================================================\n"

    # Save to src/2channel1dcnn/details.txt
    details_path_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "details.txt")
    report_content += f"\n\nMODEL ARCHITECTURE SUMMARY:\n{'-'*80}\n{model_summary_str}\n{'-'*80}\n"
    with open(details_path_1, "w") as f:
        f.write(report_content)
    print(f"=> Saved details log to {details_path_1}")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, col in enumerate(TARGET_COLUMNS):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, color="royalblue")
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title(f"{col}\nR²: {r2_scores[i]:.3f} | RMSE: {rmse_scores[i]:.3f}")
        ax.set_xlabel("True" + label_suffix)
        ax.set_ylabel("Predicted" + label_suffix)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "predicted_vs_true.png"), dpi=150)
    plt.close()

    residuals = y_pred - y_true
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=residuals, palette="Set3")
    plt.xticks(ticks=range(12), labels=TARGET_COLUMNS, rotation=45)
    plt.axhline(0, color="red", linestyle="--", alpha=0.7)
    plt.title("Distribution of Residuals across Atmospheric Species")
    plt.ylabel("Residual" + label_suffix + " (Predicted - True)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "residuals_boxplot.png"), dpi=150)
    plt.close()

    corr = np.corrcoef(residuals.T)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", xticklabels=TARGET_COLUMNS, yticklabels=TARGET_COLUMNS, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Error Correlation Matrix between Species Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "error_correlation_heatmap.png"), dpi=150)
    plt.close()

    model.train()
    mc_samples = []
    mc_loader = DataLoader(eval_subset, batch_size=20, shuffle=False)
    mc_spectra, mc_targets = next(iter(mc_loader))
    mc_spectra = mc_spectra.to(device)

    with torch.no_grad():
        for _ in range(50):
            preds = model(mc_spectra)
            mc_samples.append(preds.cpu().numpy())

    mc_predictions = np.stack(mc_samples, axis=0)
    
    # Load scaling stats and inverse-transform uncertainty samples
    mc_predictions = mc_predictions * std_y + mean_y
    true_vals = mc_targets.numpy() * std_y + mean_y
    
    mean_preds = np.mean(mc_predictions, axis=0)
    std_preds = np.std(mc_predictions, axis=0)

    key_gases = ['H2O', 'CO2', 'O2', 'CH4']
    key_indices = [TARGET_COLUMNS.index(g) for g in key_gases]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, (gas, gas_idx) in enumerate(zip(key_gases, key_indices)):
        ax = axes[idx]
        ax.errorbar(
            range(20), mean_preds[:, gas_idx], yerr=1.96 * std_preds[:, gas_idx], 
            fmt='o', color='darkblue', ecolor='lightblue', capsize=5, label="Prediction (95% CI)"
        )
        ax.scatter(range(20), true_vals[:, gas_idx], color='red', marker='x', s=80, zorder=5, label="True Value")
        ax.set_title(f"MC Dropout Uncertainty Estimates for {gas}")
        ax.set_xlabel("Test Planet Sample Index")
        ax.set_ylabel("Abundance" + label_suffix)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "mc_dropout_uncertainty.png"), dpi=150)
    plt.close()
    
    print("All evaluation plots generated successfully in charts directory.")

if __name__ == "__main__":
    evaluate()
