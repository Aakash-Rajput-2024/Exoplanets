import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset

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

def cache_data(sum_path, dir_path, save_dir, feature_mode="both"):
    dataset = InaraDataset(sum_path, dir_path, feature_mode=feature_mode)
    all_signals = []
    all_labels = []
    
    print(f"Starting to load {len(dataset)} items into memory...")
    for idx in range(len(dataset)):
        try:
            signal, label = dataset[idx]
            all_signals.append(signal)
            all_labels.append(label)
        except Exception as e:
            pass
            
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} items...")
            
    print("Converting lists to stacked tensors...")
    signals_tensor = torch.stack(all_signals)
    labels_tensor = torch.stack(all_labels)
    
    print("Performing random 80/20 split and shuffling...")
    num_valid_items = len(signals_tensor)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_valid_items, generator=generator)
    train_size = int(0.8 * num_valid_items)
    
    train_x = signals_tensor[indices[:train_size]]
    train_y = labels_tensor[indices[:train_size]]
    val_x = signals_tensor[indices[train_size:]]
    val_y = labels_tensor[indices[train_size:]]
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving train/val splits to {save_dir}...")
    torch.save(train_x, os.path.join(save_dir, "train_x.pt"))
    torch.save(train_y, os.path.join(save_dir, "train_y.pt"))
    torch.save(val_x, os.path.join(save_dir, "val_x.pt"))
    torch.save(val_y, os.path.join(save_dir, "val_y.pt"))
    print("Caching successfully completed!")

def load_cached_data(save_dir, sum_path=None, dir_path=None, normalize_inputs=True, feature_mode="planet"):
    train_x_path = os.path.join(save_dir, "train_x.pt")
    train_y_path = os.path.join(save_dir, "train_y.pt")
    val_x_path = os.path.join(save_dir, "val_x.pt")
    val_y_path = os.path.join(save_dir, "val_y.pt")
    
    if not (os.path.exists(train_x_path) and os.path.exists(train_y_path) and 
            os.path.exists(val_x_path) and os.path.exists(val_y_path)):
        if sum_path is not None and dir_path is not None:
            print(f"Cached files not found. Initiating caching process with feature_mode='{feature_mode}'...")
            cache_data(sum_path, dir_path, save_dir, feature_mode=feature_mode)
        else:
            raise FileNotFoundError(f"Cached files not found in {save_dir}")
            
    train_x = torch.load(train_x_path)
    train_y = torch.load(train_y_path)
    val_x = torch.load(val_x_path)
    val_y = torch.load(val_y_path)
    
    # Standardize inputs channel-wise
    if normalize_inputs:
        mean_x_path = os.path.join(save_dir, "mean_x.pt")
        std_x_path = os.path.join(save_dir, "std_x.pt")
        if not (os.path.exists(mean_x_path) and os.path.exists(std_x_path)):
            mean_x = train_x.mean(dim=(0, 2), keepdim=True)
            std_x = train_x.std(dim=(0, 2), keepdim=True)
            torch.save(mean_x, mean_x_path)
            torch.save(std_x, std_x_path)
        else:
            mean_x = torch.load(mean_x_path)
            std_x = torch.load(std_x_path)
            
        train_x = (train_x - mean_x) / (std_x + 1e-30)
        val_x = (val_x - mean_x) / (std_x + 1e-30)
        print("Inputs standardized channel-wise using training stats")
        
    # Standardize targets column-wise (mean 0, std 1)
    mean_y_path = os.path.join(save_dir, "mean_y.pt")
    std_y_path = os.path.join(save_dir, "std_y.pt")
    if not (os.path.exists(mean_y_path) and os.path.exists(std_y_path)):
        mean_y = train_y.mean(dim=0, keepdim=True)
        std_y = train_y.std(dim=0, keepdim=True)
        torch.save(mean_y, mean_y_path)
        torch.save(std_y, std_y_path)
    else:
        mean_y = torch.load(mean_y_path)
        std_y = torch.load(std_y_path)
        
    train_y = (train_y - mean_y) / (std_y + 1e-8)
    val_y = (val_y - mean_y) / (std_y + 1e-8)
    print("Targets standardized column-wise using training stats")
        
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)

if __name__ == "__main__":
    SUMMARY_PATH = "/Users/aakashrajput/MachineLearning/Exoplanets/data/summary.csv"
    SPECTRA_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/data/inara_1by3"
    CACHE_DIR = "/Users/aakashrajput/MachineLearning/Exoplanets/data/cache"
    
    cache_data(SUMMARY_PATH, SPECTRA_DIR, CACHE_DIR, feature_mode="both")