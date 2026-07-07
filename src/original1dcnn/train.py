import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

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

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_cached_data
from model import NasaInaraModel

BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 4
PATIENCE_ES = 10
NORMALIZE_INPUTS = True

SUMMARY_PATH = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/summary.csv"
SPECTRA_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/inara_1by3"
CACHE_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/cache_original"
CHARTS_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/original1dcnn/charts"

CHECKPOINT_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/original1dcnn/checkpoints"
LOG_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/original1dcnn/runs/inara_experiment"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    if is_best:
        torch.save(state, os.path.join(CHECKPOINT_DIR, "model_best.pth"))
    print(f"=> Saved checkpoint to {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    return start_epoch, best_loss

def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading cached dataset...")
    train_dataset, val_dataset = load_cached_data(
        CACHE_DIR, SUMMARY_PATH, SPECTRA_DIR, 
        normalize_inputs=NORMALIZE_INPUTS
    )
    
    in_channels = train_dataset[0][0].shape[0]
    seq_len = train_dataset[0][0].shape[1]
    print(f"Detected channels: {in_channels}, sequence length: {seq_len}")
    model = NasaInaraModel(in_channels=in_channels, sequence_length=seq_len).to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    # Standard L1 loss for robust standardized regression
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=LOG_DIR)

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    resume_file = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

    if os.path.isfile(resume_file):
        try:
            start_epoch, best_val_loss = load_checkpoint(resume_file, model, optimizer)
            print(f"=> Resumed training from epoch {start_epoch}")
        except Exception as e:
            print(f"=> Could not load checkpoint from {resume_file}: {e}. Starting from scratch.")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING ---
        model.train() # Activates Dropout
        train_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        for spectra, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            spectra, targets = spectra.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(spectra)
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        writer.add_scalar('Loss/Train (L1)', avg_train_loss, epoch)
        print(f"Train Loss (L1): {avg_train_loss:.4f}")

        # --- VALIDATION ---
        model.eval() # Deactivates Dropout for stable validation metrics
        val_loss = 0.0

        with torch.no_grad():
            for spectra, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                spectra, targets = spectra.to(device), targets.to(device)
                predictions = model(spectra)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        writer.add_scalar('Loss/Validation (L1)', avg_val_loss, epoch)
        print(f"Val Loss (L1): {avg_val_loss:.4f}")

        # --- EARLY STOPPING & CHECKPOINTING ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_val_loss,
        }, is_best)

        if epochs_no_improve >= PATIENCE_ES:
            print(f"\nEarly stopping triggered! No improvement for {PATIENCE_ES} epochs.")
            break

    # Save the loss curve plot at the end of training
    if len(train_loss_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label="Train Loss (L1)", marker="o")
        plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label="Val Loss (L1)", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MAE)")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(CHARTS_DIR, "loss_curve.png"), dpi=150)
        plt.close()
        print(f"=> Saved loss curve plot to {os.path.join(CHARTS_DIR, 'loss_curve.png')}")

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    train()
