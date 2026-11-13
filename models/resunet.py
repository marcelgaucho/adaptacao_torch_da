#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:43:58 2026

@author: rotunno
"""

# %% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

# %% Residual Block

class ResBlock(nn.Module):
    """Residual Block with two convolutions 3 x 3 and a skip connection, 
       as in the original paper (Zhang et al., 2017)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First stage (Pre-activation + Convolution)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # Second stage, as the first stage (Pre-activation + Convolution)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Shortcut configuration
        if stride != 1 or in_channels != out_channels: # Input != Output in resolution or channels
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else: # Input == Output in resolution and channels
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x) # get input as identity map

        # Main path with pre-activation
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # Sum with identity shortcut
        out += identity
        return out

# %% ResUnet architecture 

class ResUnet(nn.Module):
    """Complete ResUNet architecture."""
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Validate features list length
        if len(features) < 3:
            raise ValueError(f"The 'features' list must have at least 3 elements to properly structure the network in initial block, encoder, bottleneck and decoder (received: {features}).")

        # Initial convolution block with residual connection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.init_shortcut = nn.Conv2d(in_channels, features[0], kernel_size=1, bias=False)

        # Encoder (doesn't include the bottleneck)
        self.encoders = nn.ModuleList() 
        curr_channels = features[0] # Output channel size of the initial block
        
        for feature in features[1:-1]: # Build encoder blocks (default: [128, 256])
            self.encoders.append(ResBlock(curr_channels, feature, stride=2))
            curr_channels = feature
            
        # Bottleneck (Final Encoder Block)
        self.bottleneck = ResBlock(features[-2], features[-1], stride=2) # default: features[-2], features[-1] = 256, 512

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Loop through features in reverse order
        curr_channels = features[-1]  # Starts at the bottleneck output channel size (default: 512)

        for feature in features[-2::-1]: # Build decoder blocks (default: [256, 128, 64])
            # Upsample and channel reduction matching the encoder level
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # default output: [S/4, S/2, S]
                    nn.Conv2d(curr_channels, feature, kernel_size=3, padding=1, bias=False) # default output: [256, 128, 64]
                )
            )
            # Decoder block receives concatenation (feature * 2) and returns feature
            self.decoders.append(
                ResBlock(feature * 2, feature)
            )
            curr_channels = feature

        # Final output convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Initial block
        s0 = self.init_conv(x) + self.init_shortcut(x)

        # Encoder
        skip_connections = [s0] # Skip connections list begins with the output of the first block
        curr = s0
        for encoder in self.encoders:
            curr = encoder(curr) # Apply encoder
            skip_connections.append(curr) # Store skip connection

        # Bottleneck output is the deepest layer
        bottleneck_output = self.bottleneck(curr)
        
        # Decoder
        curr = bottleneck_output
        
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skip_connections)): # Loop through the decoder
            curr = upconv(curr) # Upsample + Conv
            
            # Size compatibility (in case of odd dimensions)
            if curr.shape[2:] != skip.shape[2:]:
                curr = F.interpolate(curr, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
            # Skip connection
            curr = torch.cat([curr, skip], dim=1)
            curr = decoder(curr) 

        return self.final_conv(curr)  
    
    
# %% Example

if __name__ == "__main__":
    # Instantiate ResUnet
    model = ResUnet(in_channels=3, out_channels=2, features=[64, 128, 256, 512])
    
    # Test example image (Batch size: 2, Channels: 3, Height/Width: 256x256)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")