"""
DDPM Forward Diffusion Process Implementation
Implements q(x_t|x_{t-1}) and the simple loss function
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class ForwardDiffusion:
    """
    Forward diffusion process for DDPM
    Implements q(x_t|x_0) using the reparameterization trick
    """
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        """
        Initialize the forward diffusion process with a linear noise schedule.
        
        Args:
            timesteps: Number of diffusion steps T
            beta_start: Starting value of beta (variance schedule)
            beta_end: Ending value of beta
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        # Useful for sampling q(x_t|x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior q(x_{t-1}|x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t|x_0) using the reparameterization trick.
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            x_start: Initial image [batch_size, channels, height, width]
            t: Timestep (can be a batch of timesteps) [batch_size]
            noise: Optional pre-generated noise
            
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get the schedule values for timestep t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Apply the forward diffusion formula
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a, t, x_shape):
        """
        Extract values from array a at indices t and reshape for broadcasting.
        
        Args:
            a: Array to extract from
            t: Timestep indices
            x_shape: Shape to broadcast to
            
        Returns:
            Extracted and reshaped values
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def simple_loss(model, x_start, t, forward_diffusion, noise=None):
    """
    Compute the simple loss function for DDPM (L_simple).
    
    L_simple = E[||epsilon - epsilon_theta(x_t, t)||^2]
    
    The model is trained to predict the noise epsilon that was added to x_0 to get x_t.
    
    Args:
        model: The neural network epsilon_theta(x_t, t)
        x_start: Original images [batch_size, channels, height, width]
        t: Timesteps [batch_size]
        forward_diffusion: ForwardDiffusion instance
        noise: Optional pre-generated noise
        
    Returns:
        MSE loss between predicted and actual noise
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Get noisy image at timestep t
    x_noisy = forward_diffusion.q_sample(x_start, t, noise)
    
    # Predict the noise
    predicted_noise = model(x_noisy, t)
    
    # Compute MSE loss
    loss = nn.functional.mse_loss(predicted_noise, noise)
    
    return loss
