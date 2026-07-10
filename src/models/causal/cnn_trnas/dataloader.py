import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from scipy.spatial import KDTree

class InaraDataset(Dataset):
    def __init__(self, sum_path, dir_path, feature_mode="both"):
        self.sum = pd.read_csv(sum_path, dtype={'planet_index': str})
        self.dir_path = dir_path
        self.file_names = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
        self.target_columns = [
            'H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 
            'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2'
        ]
        self.feature_mode = feature_mode
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.dir_path, file_name)
        
        planet_idx = int(file_name.split('.')[0])
        spectrum_df = pd.read_csv(file_path)
        
        # Limit to 2.0 microns
        filtering_spectra = spectrum_df[spectrum_df['wavelength_(um)'] <= 2.0]
        
        if self.feature_mode == "both":
            star_planet_signal = pd.to_numeric(filtering_spectra['star_planet_signal_(erg/s/cm2)'], errors='coerce').values
            planet_signal = pd.to_numeric(filtering_spectra['planet_signal_(erg/s/cm2)'], errors='coerce').values
            star_planet_signal = np.nan_to_num(star_planet_signal)
            planet_signal = np.nan_to_num(planet_signal)
            signal_data = np.stack([star_planet_signal, planet_signal], axis=0)
        elif self.feature_mode == "planet":
            planet_signal = pd.to_numeric(filtering_spectra['planet_signal_(erg/s/cm2)'], errors='coerce').values
            planet_signal = np.nan_to_num(planet_signal)
            signal_data = np.expand_dims(planet_signal, axis=0)
        else: # "star_planet"
            star_planet_signal = pd.to_numeric(filtering_spectra['star_planet_signal_(erg/s/cm2)'], errors='coerce').values
            star_planet_signal = np.nan_to_num(star_planet_signal)
            signal_data = np.expand_dims(star_planet_signal, axis=0)
            
        metadata_row = self.sum[self.sum['planet_index'].astype(int) == planet_idx]
        if len(metadata_row) == 0:
            raise ValueError(f"Uh oh! Could not find planet ID {planet_idx} in summary.csv")
        
        labels = metadata_row[self.target_columns].values[0]
        signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return signal_tensor, label_tensor

def load_cached_data_with_envs(save_dir, sum_path, normalize_inputs=True):
    # Load standardized cached splits
    train_x = torch.load(os.path.join(save_dir, "train_x.pt"))
    train_y = torch.load(os.path.join(save_dir, "train_y.pt"))
    val_x = torch.load(os.path.join(save_dir, "val_x.pt"))
    val_y = torch.load(os.path.join(save_dir, "val_y.pt"))
    
    # Load scaling stats for inputs/targets
    mean_x = torch.load(os.path.join(save_dir, "mean_x.pt"))
    std_x = torch.load(os.path.join(save_dir, "std_x.pt"))
    
    mean_y = torch.load(os.path.join(save_dir, "mean_y.pt"))
    std_y = torch.load(os.path.join(save_dir, "std_y.pt"))
    
    # Standardize inputs channel-wise
    if normalize_inputs:
        train_x_std = (train_x - mean_x) / (std_x + 1e-30)
        val_x_std = (val_x - mean_x) / (std_x + 1e-30)
    else:
        train_x_std = train_x
        val_x_std = val_x
        
    train_y_std = (train_y - mean_y) / (std_y + 1e-8)
    val_y_std = (val_y - mean_y) / (std_y + 1e-8)
    
    # Load summary and align envs using KDTree matching
    summary = pd.read_csv(sum_path, dtype={"planet_index": str})
    summary["planet_index_int"] = summary["planet_index"].astype(int)
    summary = summary.set_index("planet_index_int")
    
    target_columns = [
        'H2O', 'CO2', 'O2', 'N2', 'CH4', 'N2O', 
        'CO', 'O3', 'SO2', 'NH3', 'C2H6', 'NO2'
    ]
    summary_targets = summary[target_columns].values
    tree = KDTree(summary_targets)
    
    # Find nearest neighbor for each sample's targets (y). Cached splits do not
    # store planet_ids, so we recover environment rows via NN on the label vector.
    # This is brittle: if two distinct planets have near-identical compositions
    # but different (star, env) parameters, env conditioning gets contaminated.
    train_dist_nn, train_idx_nn = tree.query(train_y.numpy(), k=1)
    val_dist_nn, val_idx_nn = tree.query(val_y.numpy(), k=1)

    nn_threshold = 1e-4
    n_train_bad = int((train_dist_nn > nn_threshold).sum())
    n_val_bad = int((val_dist_nn > nn_threshold).sum())
    print(
        f"[KDTree match] train NN distance: "
        f"median={np.median(train_dist_nn):.2e} max={train_dist_nn.max():.2e} | "
        f"val: median={np.median(val_dist_nn):.2e} max={val_dist_nn.max():.2e}"
    )
    if n_train_bad or n_val_bad:
        print(
            f"[WARN] {n_train_bad}/{len(train_dist_nn)} train and "
            f"{n_val_bad}/{len(val_dist_nn)} val samples matched with NN "
            f"distance > {nn_threshold:g}; env conditioning may be noisy. "
            f"Consider caching planet_ids alongside the splits."
        )

    train_planet_ids = summary.index[train_idx_nn].tolist()
    val_planet_ids = summary.index[val_idx_nn].tolist()
    
    train_rows = summary.loc[train_planet_ids]
    val_rows = summary.loc[val_planet_ids]
    
    # Continuous environment columns
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
    
    env_stats = {
        "means_e": torch.tensor(means_e.values, dtype=torch.float32),
        "stds_e": torch.tensor(stds_e.values, dtype=torch.float32),
        "class_to_idx": class_to_idx,
        "unique_classes": unique_classes,
        "cont_cols": cont_cols
    }
    
    return (TensorDataset(train_x_std, train_y_std, train_env_tensor), 
            TensorDataset(val_x_std, val_y_std, val_env_tensor),
            env_stats)
