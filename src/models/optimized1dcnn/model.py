import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchsummary import summary
except ImportError:
    summary = None


def _mps_safe_adaptive_avg_pool1d(x, output_size):
    """AdaptiveAvgPool1d that also works on Apple MPS.

    MPS has no kernel for adaptive pooling when the input length is not divisible
    by the output size (pytorch#96056) — it raises instead of falling back. Here
    the post-conv length is 4379→547 (prime) and output_size=16, so on MPS we run
    just this one op on CPU. The result is numerically identical to the CPU/CUDA
    path; only this pooling call takes the detour, and only on MPS.
    """
    if x.device.type == "mps" and x.shape[-1] % output_size != 0:
        return F.adaptive_avg_pool1d(x.cpu(), output_size).to(x.device)
    return F.adaptive_avg_pool1d(x, output_size)

class NasaInaraModel(nn.Module):
    def __init__(self, in_channels=1, sequence_length=4379):
        super().__init__()
        
        # 1D CNN Architecture with progressive kernels and BatchNorm for stability
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        self.flatten = nn.Flatten()
        
        # Dynamically calculate flattened dimension using a dummy forward pass
        dummy_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            x = self.pool1(self.tanh(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu(self.bn3(self.conv3(x))))
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.adaptive_pool(x)
            x = self.flatten(x)
            self.nodes = x.shape[1]
            
        print(f"Dynamically determined model input features: {sequence_length}, channels: {in_channels}, flat nodes: {self.nodes}")
        
        # Dense layers with Dropout for Monte Carlo approximation
        self.fc1 = nn.Linear(self.nodes, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 12)
        
    def forward(self, x):
        x = self.pool1(self.tanh(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = _mps_safe_adaptive_avg_pool1d(x, 16)
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

if __name__ == "__main__":
    dummy_input = torch.randn(32, 2, 4379) 
    model = NasaInaraModel(in_channels=2, sequence_length=4379)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")