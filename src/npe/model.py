import torch
import torch.nn as nn
import math
from sbi.neural_nets import posterior_nn

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

class TransformerSummaryNetwork(nn.Module):
    """
    Summary network that takes the spectrum and embeds it into a fixed length 
    representation for the Normalizing Flow.
    """
    def __init__(self, in_channels=1, sequence_length=4379, d_model=128, n_heads=8, num_layers=2):
        super().__init__()
        
        # 1D CNN Downsampling Block
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
        
        # Calc length
        dummy_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            x = self.pool1(self.tanh(self.conv1(dummy_input)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            x = self.relu_conv(self.conv4(x))
            self.downsampled_len = x.shape[2]
            
        self.pre_transformer_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.downsampled_len + 100)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x is [batch_size, seq_len] or [batch_size, in_channels, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dim
            
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.relu_conv(self.conv4(x))
        
        x = x.permute(0, 2, 1) 
        x = self.pre_transformer_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        x = self.post_norm(x)
        return x

def build_neural_posterior(summary_net, model_type='nsf', hidden_features=64, num_transforms=5):
    """
    Builds the Neural Posterior using Neural Spline Flows (NSF)
    """
    neural_posterior = posterior_nn(
        model=model_type, 
        embedding_net=summary_net, 
        hidden_features=hidden_features, 
        num_transforms=num_transforms
    )
    return neural_posterior
