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
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Initial convolution with residual connection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1, bias=False)
        )
        self.init_shortcut = nn.Conv2d(in_channels, features[0], kernel_size=1, bias=False)

        # Encoder
        self.encoders = nn.ModuleList()
        curr_channels = features[0]
        for feature in features[1:]:
            self.encoders.append(ResBlock(curr_channels, feature, stride=2))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = ResBlock(features[-1], features[-1] * 2, stride=2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        rev_features = features[::-1]
        curr_channels = features[-1] * 2

        for feature in rev_features:
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(curr_channels, feature, kernel_size=3, padding=1, bias=False)
                )
            )
            self.decoders.append(
                ResBlock(feature * 2, feature)
            )
            curr_channels = feature

        # Final output convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Initial block
        init_out = self.init_conv(x)
        identity = self.init_shortcut(x)
        s0 = init_out + identity

        # Encoder
        skip_connections = [s0]
        curr = s0
        for encoder in self.encoders:
            curr = encoder(curr)
            skip_connections.append(curr)

        # Bottleneck
        curr = self.bottleneck(curr)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            curr = upconv(curr)
            skip = skip_connections[i]
            
            # Size compatibility (in case of odd dimensions)
            if curr.shape != skip.shape:
                curr = F.interpolate(curr, size=skip.shape[2:], mode='bilinear', align_corners=False)
                
            # Skip connection
            curr = torch.cat([curr, skip], dim=1)
            curr = decoder(curr) 

        return self.final_conv(curr)       

# %% Example

if __name__ == "__main__":
    # Instantiate ResUnet
    model = ResUnet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    
    # Test example image (Batch size: 2, Channels: 3, Height/Width: 256x256)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")