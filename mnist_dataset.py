#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:32:00 2026

@author: rotunno
"""

# %% Imports

# Standard imports
import random

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision.datasets as datasets

# Other imports
import matplotlib.pyplot as plt
plt.ioff() # Plot only when plt.show is called

# %%

class MNISTSegmentationDataset(Dataset):
    """
    Custom Semantic Segmentation Dataset for Domain Adaptation based on MNIST.
    Generate synthetic canvas images by placing random MNIST digits, while creating 
    pixel-wise ground-truth segmentation masks.   
    """
    def __init__(self, mnist_dataset, canvas_size=(64, 64), num_digits_per_image=3, domain="source"):
        self.mnist_dataset = mnist_dataset 
        self.canvas_size = canvas_size
        self.num_digits_per_image = num_digits_per_image
        self.domain = domain # "source" or "target"
        
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        H, W = self.canvas_size
        
        # Initialize empty canvas (1 channel, H, W) and segmentation mask (H, W)
        # Background class is represented by 0
        canvas = torch.zeros((1, H, W), dtype=torch.float32)
        mask = torch.zeros((H, W), dtype=torch.long)
        
        d_h, d_w = 28, 28 # Original MNIST digit size
        
        # Place random digits onto the canvas
        for _ in range(self.num_digits_per_image):
            # Pick a random sample from the MNIST dataset
            rand_idx = random.randint(0, len(self.mnist_dataset) - 1)
            digit_img, digit_label = self.mnist_dataset[rand_idx]
            
            # Generate random coordinates for placement
            top = random.randint(0, H - d_h)
            left = random.randint(0, W - d_w)
            
            # Paste the digit onto the canvas (using max to handle overlapping pixels)
            canvas[:, top:top+d_h, left:left+d_w] = torch.maximum(
                canvas[:, top:top+d_h, left:left+d_w], 
                digit_img
            )
            
            # Create binary mask for the current digit
            digit_mask = (digit_img.squeeze(dim=0) > 0.1).long()
            
            # Assign class labels to the segmentation mask (Label + 1, since 0 is background)
            sub_mask = mask[top:top+d_h, left:left+d_w]
            sub_mask[digit_mask == 1] = digit_label + 1 # sub_mask is a view, so mask is already changed
            
        # Apply domain-specific transformations to simulate domain shift
        if self.domain == "target":
            # Domain shift: Add Gaussian noise and invert background intensities
            noise = torch.randn_like(canvas) * 0.15
            canvas = torch.clamp(canvas + noise, 0.0, 1.0)
            
            # Optional: Invert or shift intensity for target domain style
            # canvas = 1.0 - canvas
            
        return canvas, mask
    
# %%

def plot_data(src_dataset, tgt_dataset, sample_idx):
    """
    Plot image and masks side by side, from source and target domains.
    """
    # Get image and masks for a sample from source and target domains
    img_source, mask_source = src_dataset[sample_idx]
    img_target, mask_target = tgt_dataset[sample_idx]
    
    # Create a figure with 2 rows (source and target), each containing image and mask
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Row 1: Source Domain (Clean)
    axes[0, 0].imshow(img_source.squeeze(0).cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Source Canvas (Clean)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_source.cpu().numpy(), cmap='tab10', interpolation='nearest', vmin=0, vmax=10)
    axes[0, 1].set_title("Source Mask (Ground Truth)")
    axes[0, 1].axis('off')
    
    # Row 2: Target Domain (Noisy)
    axes[1, 0].imshow(img_target.squeeze(0).cpu().numpy(), cmap='gray')
    axes[1, 0].set_title("Target Canvas (Noisy)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_target.cpu().numpy(), cmap='tab10', interpolation='nearest', vmin=0, vmax=10)
    axes[1, 1].set_title("Target Mask (Ground Truth)")
    axes[1, 1].axis('off')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
# %%

# 1. Download base MNIST dataset
transform = v2.Compose([
    v2.ToImage(),                  # Converts PIL Image to Tensor keeping uint8 (0-255)
    v2.ToDtype(torch.float32, scale=True)  # Converts to float32 and scales to [0.0, 1.0] range
])

mnist_train = datasets.MNIST(root='./torch_data', train=True, download=True, transform=transform)

# 2. Create Source Domain Dataset (Clean background)
source_dataset = MNISTSegmentationDataset(
        mnist_dataset=mnist_train,
        canvas_size=(64, 64),
        num_digits_per_image=3,
        domain="source"   
    )

# 3. Create Target Domain Dataset (Noisy/Shifted background)
target_dataset = MNISTSegmentationDataset(
        mnist_dataset=mnist_train,
        canvas_size=(64, 64),
        num_digits_per_image=3,
        domain="target"   
    )

# 4. View datasets
# Plot a sample from source and target domains
plot_data(source_dataset, target_dataset, sample_idx=0)


# 5. Setup DataLoaders for training
source_loader = DataLoader(source_dataset, batch_size=16, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)

# Quick validation check of batch shapes
for images, masks in source_loader:
    print(f"Source Images batch shape: {images.shape}")  # Expected: [16, 1, 64, 64]
    print(f"Source Masks batch shape:  {masks.shape}")   # Expected: [16, 64, 64]
    break



