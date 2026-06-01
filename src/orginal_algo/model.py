import torch
import torch.nn as nn

class NasaInaraModel(nn.Module):
    def __init__(self, in_channels=1, sequence_length=4379):
        super().__init__()
        
        # 1D CNN Architecture as defined in the paper
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        
        # Dynamically calculate flattened dimension using a dummy forward pass
        dummy_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            x = self.pool1(self.tanh(self.conv1(dummy_input)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            x = self.relu(self.conv4(x))
            x = self.flatten(x)
            self.nodes = x.shape[1]
            
        print(f"Dynamically determined model input features: {sequence_length}, channels: {in_channels}, flat nodes: {self.nodes}")
        
        # Dense layers with Dropout for Monte Carlo approximation
        self.fc1 = nn.Linear(self.nodes, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 12)
        
    def forward(self, x):
        x = self.pool1(self.tanh(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

if __name__ == "__main__":
    dummy_input = torch.randn(32, 2, 4379) 
    model = NasaInaraModel(in_channels=2, sequence_length=4379)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")