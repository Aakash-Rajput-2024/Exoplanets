import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
sys.path.insert(0, THIS_DIR)                        # local dataloader.py / model.py
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))  # shared 'common' package

from dataloader import load_cached_data
from model import NasaInaraTransformer
from common.provenance import stamp_checkpoint

BATCH_SIZE = 16
LEARNING_RATE = 0.001       # Higher peak LR (warmup makes this safe)
EPOCHS = 50                 # Transformers need more epochs to converge
PATIENCE_ES = 15            # More patience for transformer convergence
WARMUP_EPOCHS = 5           # Linear warmup before cosine decay
NORMALIZE_INPUTS = True

# All paths derived from this file's location so the training script and its
# evaluator (test.py) agree on where checkpoints live by construction (fixes C6:
# the previous CHECKPOINT_DIR pointed into a deleted src/10_mil_minus_params/ tree).
SUMMARY_PATH = os.path.join(REPO_ROOT, "data", "summary.csv")
SPECTRA_DIR = os.path.join(REPO_ROOT, "data", "inara_1by3")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache_planet")
CHARTS_DIR = os.path.join(THIS_DIR, "charts")
CHECKPOINT_DIR = os.path.join(THIS_DIR, "checkpoints")
LOG_DIR = os.path.join(THIS_DIR, "runs", "inara_experiment")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Hyperparameter fingerprint embedded in every checkpoint for traceable
# provenance (fixes C6). Runtime shape (in/out channels, seq_len) is added in
# train() once the data is loaded.
CONFIG = dict(
    track="transformerarch", model="NasaInaraTransformer",
    batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS,
    patience_es=PATIENCE_ES, warmup_epochs=WARMUP_EPOCHS,
    normalize_inputs=NORMALIZE_INPUTS, loss="mse", optimizer="adam",
    scheduler="cosine_warmup", grad_clip=1.0, feature_mode="planet",
    cache="cache_planet",
)

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warmup for warmup_epochs, then cosine decay to 0."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay: 1 → 0
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

    print("Loading cached dataset...")
    train_dataset, val_dataset = load_cached_data(
        CACHE_DIR, SUMMARY_PATH, SPECTRA_DIR, 
        normalize_inputs=NORMALIZE_INPUTS,
        feature_mode="planet"
    )
    
    in_channels = train_dataset[0][0].shape[0]
    seq_len = train_dataset[0][0].shape[1]
    print(f"Detected channels: {in_channels}, sequence length: {seq_len}")
    model = NasaInaraTransformer(in_channels=in_channels, sequence_length=seq_len).to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    # MSE loss — smoother gradients than L1, better for transformer optimization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Cosine annealing with linear warmup
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
            print(f"=> Could not load checkpoint from {resume_file}: {e}. Starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING ---
        model.train()  # Activates Dropout
        train_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        for spectra, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            spectra, targets = spectra.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(spectra)
            
            loss = criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping — prevents explosion through attention layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Step per batch for warmup scheduler

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        train_loss_history.append(avg_train_loss)
        lr_history.append(current_lr)
        writer.add_scalar('Loss/Train (MSE)', avg_train_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        print(f"Train Loss (MSE): {avg_train_loss:.6f} | LR: {current_lr:.6f}")

        # --- VALIDATION ---
        model.eval()  # Deactivates Dropout for stable validation metrics
        val_loss = 0.0

        with torch.no_grad():
            for spectra, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                spectra, targets = spectra.to(device), targets.to(device)
                predictions = model(spectra)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        writer.add_scalar('Loss/Validation (MSE)', avg_val_loss, epoch)
        print(f"Val Loss (MSE): {avg_val_loss:.6f}")

        # --- EARLY STOPPING & CHECKPOINTING ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        state = stamp_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'lr_history': lr_history,
            'epochs_no_improve': epochs_no_improve,
        }, {**CONFIG, "in_channels": in_channels, "seq_len": seq_len}, repo_dir=REPO_ROOT)
        save_checkpoint(state, is_best)

        if epochs_no_improve >= PATIENCE_ES:
            print(f"\nEarly stopping triggered! No improvement for {PATIENCE_ES} epochs.")
            break

    # Save the loss curve plot at the end of training
    if len(train_loss_history) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss curve
        ax1.plot(range(1, len(train_loss_history)+1), train_loss_history, label="Train Loss (MSE)", marker="o")
        ax1.plot(range(1, len(val_loss_history)+1), val_loss_history, label="Val Loss (MSE)", marker="o")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Training and Validation Loss Curve")
        ax1.legend()
        ax1.grid(True)
        
        # LR schedule
        ax2.plot(range(1, len(lr_history)+1), lr_history, label="Learning Rate", color="green", marker="o")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule (Warmup + Cosine)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, "loss_curve.png"), dpi=150)
        plt.close()
        print(f"=> Saved loss curve plot to {os.path.join(CHARTS_DIR, 'loss_curve.png')}")

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    train()
