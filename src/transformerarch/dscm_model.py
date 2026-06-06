import torch
import torch.nn as nn

class DSCMEncoder(nn.Module):
    def __init__(self, in_channels=1, sequence_length=4379, a_dim=12, e_dim=8, latent_dim=64):
        super().__init__()
        # 1D CNN Downsampling Block (mirroring NasaInaraTransformer)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu_conv = nn.ReLU()
        
        # Calculate downsampled sequence length
        dummy_input = torch.randn(1, in_channels, sequence_length)
        with torch.no_grad():
            x = self.pool1(self.tanh(self.conv1(dummy_input)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            x = self.relu_conv(self.conv4(x))
            self.downsampled_len = x.shape[2]
        
        # Global Average Pooling to reduce sequence dimension
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # MLP to predict mu and logvar
        input_mlp_dim = 128 + a_dim + e_dim
        self.fc = nn.Sequential(
            nn.Linear(input_mlp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, s, a, e):
        # s: [B, 1, 4379]
        # a: [B, a_dim]
        # e: [B, e_dim]
        x = self.pool1(self.tanh(self.conv1(s)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.relu_conv(self.conv4(x))
        
        x = self.gap(x).squeeze(-1) # [B, 128]
        
        # Concatenate features with conditioned variables A and E
        x_cond = torch.cat([x, a, e], dim=-1)
        
        h = self.fc(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class DSCMDecoder(nn.Module):
    def __init__(self, sequence_length=4379, a_dim=12, e_dim=8, latent_dim=64):
        super().__init__()
        input_dim = latent_dim + a_dim + e_dim
        
        # Project conditioning + latent representation to bottleneck shape
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 128),
            nn.ReLU()
        )
        
        # Sequential Upsampling and Convolutions
        self.up1 = nn.Upsample(size=512, mode='linear', align_corners=True)
        self.conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.up2 = nn.Upsample(size=2048, mode='linear', align_corners=True)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.up3 = nn.Upsample(size=sequence_length, mode='linear', align_corners=True)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, z, a, e):
        # z: [B, latent_dim]
        # a: [B, a_dim]
        # e: [B, e_dim]
        x_cond = torch.cat([z, a, e], dim=-1)
        h = self.fc(x_cond)
        h = h.view(-1, 64, 128) # [B, 64, 128]
        
        x = self.relu1(self.conv1(self.up1(h)))
        x = self.relu2(self.conv2(self.up2(x)))
        x = self.conv3(self.up3(x)) # [B, 1, sequence_length]
        return x

class DSCM(nn.Module):
    def __init__(self, sequence_length=4379, a_dim=12, e_dim=8, latent_dim=64):
        super().__init__()
        self.encoder = DSCMEncoder(sequence_length=sequence_length, a_dim=a_dim, e_dim=e_dim, latent_dim=latent_dim)
        self.decoder = DSCMDecoder(sequence_length=sequence_length, a_dim=a_dim, e_dim=e_dim, latent_dim=latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, s, a, e):
        mu, logvar = self.encoder(s, a, e)
        z = self.reparameterize(mu, logvar)
        recon_s = self.decoder(z, a, e)
        return recon_s, mu, logvar
