import torch
import torch.nn as nn
import math

try:
    from torchinfo import summary
except ImportError:
    summary = None

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
        
        # 1. 1D CNN Downsampling Block — 3 pools for 8x downsampling (matching CNN baseline)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Clean 2x downsample
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # Clean 2x downsample
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # Third pool — was missing before!
        
        self.conv4 = nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        self.relu_conv = nn.ReLU()
        
        # Calculate the downsampled sequence length using a dummy forward pass
        dummy_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            x = self.pool1(self.tanh(self.conv1(dummy_input)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            x = self.relu_conv(self.conv4(x))
            self.downsampled_len = x.shape[2]
            
        print(f"Dynamically determined downsampled sequence length: {self.downsampled_len}")
        
        # 2. LayerNorm before Transformer (stabilizes training)
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
        
        # 4. Global Average Pooling + Small MLP Head (replaces flatten+massive FC bottleneck)
        # GAP: [B, seq_len, d_model] → mean over seq → [B, d_model]
        # This eliminates the 70K→500 bottleneck that caused mean collapse
        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # 1. Conv Downsampling Block
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.relu_conv(self.conv4(x))
        
        # 2. Permute to Transformer Input Shape [Batch, SeqLen, d_model]
        x = x.permute(0, 2, 1) 
        x = self.pre_transformer_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 3. Global Average Pooling over the sequence dimension
        x = x.mean(dim=1)  # [B, seq_len, d_model] → [B, d_model]
        x = self.post_norm(x)
        
        # 4. MLP Classification Head
        out = self.head(x)
        return out

if __name__ == "__main__":
    dummy_input = torch.randn(32, 1, 4379) 
    model = NasaInaraTransformer(in_channels=1, sequence_length=4379)
    output = model(dummy_input)
    
    print("-" * 50)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("-" * 50)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"MLP Head Parameters: {head_params:,}")
    print("-" * 50)
    
    if summary:
        print("\nModel Summary:")
        summary(model, input_size=(32, 1, 4379), device="cpu")
    else:
        print("Install torchinfo ('pip install torchinfo') to view the detailed layer summary.")