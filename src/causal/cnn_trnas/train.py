import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
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

from dataloader import load_cached_data_with_envs
from model import NasaInaraTransformer

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE_ES = 15
WARMUP_EPOCHS = 5

SUMMARY_PATH = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/summary.csv"
CACHE_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/data/cache_planet"
CHECKPOINT_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/causal/cnn_trnas/checkpoints"
CHARTS_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/causal/cnn_trnas/charts"
LOG_DIR = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src/causal/cnn_trnas/runs/causal_experiment"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    if is_best:
        torch.save(state, os.path.join(CHECKPOINT_DIR, "model_best.pth"))
    print(f"=> Saved checkpoint to {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded scheduler state")
        
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    lr_history = checkpoint.get('lr_history', [])
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    
    return start_epoch, best_loss, train_loss_history, val_loss_history, lr_history, epochs_no_improve

def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Load counterfactually augmented training data
    aug_x_path = os.path.join(CHECKPOINT_DIR, "train_x_augmented.pt")
    aug_y_path = os.path.join(CHECKPOINT_DIR, "train_y_augmented.pt")
    
    if not (os.path.exists(aug_x_path) and os.path.exists(aug_y_path)):
        raise FileNotFoundError("Augmented train datasets not found! Run generate_counterfactuals.py first.")
        
    print("Loading augmented training dataset...")
    train_x = torch.load(aug_x_path)
    train_y = torch.load(aug_y_path)
    train_dataset = TensorDataset(train_x, train_y)
    
    # 2. Load validation dataset
    _, val_dataset, _ = load_cached_data_with_envs(CACHE_DIR, SUMMARY_PATH)
    val_x = val_dataset.tensors[0]
    val_y = val_dataset.tensors[1]
    val_dataset = TensorDataset(val_x, val_y)
    
    in_channels = train_x.shape[1]
    seq_len = train_x.shape[2]
    print(f"Detected channels: {in_channels}, sequence length: {seq_len}")
    
    model = NasaInaraTransformer(in_channels=in_channels, sequence_length=seq_len).to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"Loaded {len(train_dataset)} training (augmented) and {len(val_dataset)} validation samples.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = get_cosine_warmup_scheduler(
        optimizer, 
        warmup_epochs=WARMUP_EPOCHS, 
        total_epochs=EPOCHS, 
        steps_per_epoch=len(train_loader)
    )
    
    writer = SummaryWriter(log_dir=LOG_DIR)

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loss_history = []
    val_loss_history = []
    lr_history = []

    resume_file = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    if os.path.isfile(resume_file):
        try:
            (start_epoch, 
             best_val_loss, 
             train_loss_history, 
             val_loss_history, 
             lr_history, 
             epochs_no_improve) = load_checkpoint(resume_file, model, optimizer, scheduler)
            print(f"=> Resumed training from epoch {start_epoch}")
        except Exception as e:
            print(f"=> Could not load checkpoint: {e}. Starting from scratch.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        for spectra, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            spectra, targets = spectra.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(spectra)
            
            loss = criterion(predictions, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        train_loss_history.append(avg_train_loss)
        lr_history.append(current_lr)
        writer.add_scalar('Loss/Train (MSE)', avg_train_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        print(f"Train Loss (MSE): {avg_train_loss:.6f} | LR: {current_lr:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for spectra, targets in val_loader:
                spectra, targets = spectra.to(device), targets.to(device)
                predictions = model(spectra)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        writer.add_scalar('Loss/Validation (MSE)', avg_val_loss, epoch)
        print(f"Val Loss (MSE): {avg_val_loss:.6f}")

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
            'scheduler': scheduler.state_dict(),
            'best_loss': best_val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'lr_history': lr_history,
            'epochs_no_improve': epochs_no_improve,
        }, is_best)

        if epochs_no_improve >= PATIENCE_ES:
            print(f"\nEarly stopping triggered! No improvement for {PATIENCE_ES} epochs.")
            break

    if len(train_loss_history) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(range(1, len(train_loss_history)+1), train_loss_history, label="Train Loss", marker="o")
        ax1.plot(range(1, len(val_loss_history)+1), val_loss_history, label="Val Loss", marker="o")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(range(1, len(lr_history)+1), lr_history, label="Learning Rate", color="green", marker="o")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, "loss_curve.png"), dpi=150)
        plt.close()
        print("Saved loss curves.")

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    train()
