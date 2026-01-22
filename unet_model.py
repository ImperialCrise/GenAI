import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for timestep conditioning.
    Converts timestep t into a rich embedding vector.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: Timestep tensor [batch_size]
        Returns:
            Embedded timestep [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    Applies convolutions and reduces spatial dimensions.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.MaxPool2d(2)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, t_emb):
        """
        Args:
            x: Input tensor [batch, in_channels, H, W]
            t_emb: Time embedding [batch, time_emb_dim]
        Returns:
            downsampled: Downsampled output [batch, out_channels, H/2, W/2]
            skip: Skip connection [batch, out_channels, H, W]
        """
        # First convolution
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]
        
        # Second convolution
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        
        # Store for skip connection
        skip = h
        
        # Downsample
        downsampled = self.downsample(h)
        
        return downsampled, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder.
    Applies transposed convolutions and increases spatial dimensions.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, skip, t_emb):
        """
        Args:
            x: Input tensor [batch, in_channels, H, W]
            skip: Skip connection from encoder [batch, out_channels, H*2, W*2]
            t_emb: Time embedding [batch, time_emb_dim]
        Returns:
            Output tensor [batch, out_channels, H*2, W*2]
        """
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Add time embedding
        time_emb = self.time_mlp(t_emb)
        x = x + time_emb[:, :, None, None]
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net model for DDPM noise prediction.
    Architecture: epsilon_theta(x_t, t) -> epsilon
    
    Takes a noisy image x_t and timestep t as input,
    outputs the predicted noise epsilon.
    """
    
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128, base_channels=64):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale MNIST)
            out_channels: Number of output channels (1 for grayscale)
            time_emb_dim: Dimension of time embedding
            base_channels: Base number of channels (will be multiplied in deeper layers)
        """
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Time embedding for bottleneck
        self.bottleneck_time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, base_channels * 8),
            nn.ReLU()
        )
        
        # Decoder (upsampling path)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        
        # Final convolution to get back to image space
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x, t):
        """
        Forward pass of the U-Net.
        
        Args:
            x: Noisy image [batch_size, channels, H, W]
            t: Timestep [batch_size]
            
        Returns:
            Predicted noise [batch_size, channels, H, W]
        """
        # Get time embedding
        t_emb = self.time_embedding(t)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        
        # Bottleneck
        x = self.bottleneck(x)
        # Add time embedding to bottleneck
        time_emb = self.bottleneck_time_mlp(t_emb)
        x = x + time_emb[:, :, None, None]
        
        # Decoder
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        
        # Final convolution to predict noise
        noise_pred = self.final_conv(x)
        
        return noise_pred
    
    def get_num_parameters(self):
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device='cuda', base_channels=64):

    model = UNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        base_channels=base_channels
    ).to(device)
    
    return model