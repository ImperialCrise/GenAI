import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from ddpm_forward import ForwardDiffusion, simple_loss
from unet_model import create_model


def train_epoch(model, dataloader, optimizer, forward_diffusion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: U-Net model
        dataloader: Training data loader
        optimizer: Optimizer
        forward_diffusion: ForwardDiffusion instance
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps for each image in the batch
        t = torch.randint(0, forward_diffusion.timesteps, (batch_size,), device=device)
        
        # Compute loss
        loss = simple_loss(model, images, t, forward_diffusion)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, forward_diffusion, device):
    """
    Validate the model.
    
    Args:
        model: U-Net model
        dataloader: Validation data loader
        forward_diffusion: ForwardDiffusion instance
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, forward_diffusion.timesteps, (batch_size,), device=device)
            
            # Compute loss
            loss = simple_loss(model, images, t, forward_diffusion)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss