import torch
import torch.nn as nn
import math

# NOTE: the legacy VAE-DSCM classes (DSCMEncoder/DSCMDecoder/DSCM) that used to
# live here were removed. They were unused by the v2 pipeline (registry.py loads
# only NasaInaraTransformer by path) and had DIVERGED from the copy in
# src/transformerarch/dscm_model.py -- same class names/state_dict shape, but
# different hardcoded decoder upsample sizes (512/2048 there vs. seq-derived
# 547/2189 here), i.e. two silently-different forward() implementations sharing
# a name. The do-calculus counterfactual track now uses exact environment
# re-pairing instead of a learned VAE (see src/common/counterfactuals.py for the
# replacement and the full rationale). Removing this eliminates the divergence
# rather than picking a "winner" between two already-abandoned implementations.
# train_dscm.py / smoke_test.py / generate_counterfactuals.py in this directory
# import these removed classes and are legacy/dead code as a result -- they were
# already unreachable from the v2 pipeline (registry.py, train_runner.py) before
# this change.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class NasaInaraTransformer(nn.Module):
    def __init__(self, in_channels=1, sequence_length=4379, d_model=128, n_heads=8, num_layers=2, output_dim=12):
        super().__init__()
        
        # 1. 1D CNN Downsampling Block
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.conv4 = nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        self.relu_conv = nn.ReLU()
        
        # Calculate the downsampled sequence length
        dummy_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            x = self.pool1(self.tanh(self.conv1(dummy_input)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            x = self.relu_conv(self.conv4(x))
            self.downsampled_len = x.shape[2]
            
        # 2. LayerNorm before Transformer
        self.pre_transformer_norm = nn.LayerNorm(d_model)
        
        # 3. Positional Encoding and Transformer Encoder Layer
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.downsampled_len + 100)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Global Average Pooling + Small MLP Head
        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.relu_conv(self.conv4(x))
        
        x = x.permute(0, 2, 1) 
        x = self.pre_transformer_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        x = self.post_norm(x)
        out = self.head(x)
        return out
