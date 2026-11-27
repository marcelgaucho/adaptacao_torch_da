# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:32:11 2026

@author: Marcel
"""

# %% Imports

from pathlib import Path
from osgeo import gdal, ogr, osr
import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view

# %%

class PatchExtractor:
    def __init__(self, patch_size=256, 
                 overlap_percent=0.25, padding=False):
        if not (0 <= overlap_percent < 1.0): raise ValueError("overlap_percent must be >= 0 and < 1")
            
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.overlap_px = int(patch_size * overlap_percent)
        
        self.stride = self.patch_size - self.overlap_px
        
        # We always move forward
        if self.stride < 1:
            raise ValueError('Stride must be greater than or equal to 1. Check overlap_percent value.')
        
        self.padding = padding
        
    def __call__(self, array, get_coords=True, pad_value=None):
        """
        array: 3D array H, W, C]
        Returns patches (N, S, S, C) and coords (with get_coords==True) (N, 4) = [ymin, xmin, ymax, xmax], relatives to the original array (can be negative with padding).
        """
        if len(array.shape) != 3:
            raise ValueError('Image must be in shape (Height, Width, Channels)') # Image shape restriction

        h, w, c = array.shape # tile dimenstions
        
        S, stride = self.patch_size, self.stride # patch size and stride for extraction
        
        # Symmetric padding
        if self.padding:
            # 1. Calculate 'SAME' Padding (TensorFlow style)
            out_h = math.ceil(h / self.stride)
            out_w = math.ceil(w / self.stride)
            
            # Get how much it overflowed the border (if it happens)
            pad_h_total = max(0, (out_h - 1) * self.stride + self.patch_size - h)
            pad_w_total = max(0, (out_w - 1) * self.stride + self.patch_size - w)
            
            # If odd, bottom and right stay with 1 more pixel
            top, left = pad_h_total // 2, pad_w_total // 2
            bottom, right = pad_h_total - top, pad_w_total - left
            pad_width = ((top, bottom), (left, right), (0,0))
            
            # 2. Apply Symmetric Padding
            if pad_value is None: # default is symmetric padding
                array = np.pad(array, pad_width, mode='symmetric')
            else: # constant padding if value is passed
                array = np.pad(array, pad_width, mode="constant", constant_values=pad_value)
            
            # Global pixel offset (relative to original top-left 0,0)
            # Starting at -top/-left because the patch window starts "outside" the original image
            start_y, start_x = -top, -left
        # Only valid pixels
        else:
            if h < S or w < S:
                raise ValueError(f"Tile ({h}x{w}) is smaller than patch size ({S}); use padding=True") # patch is greater than tile case
                
            # VALID: No padding, calculate how many full patches fit
            out_h = (h - self.patch_size) // self.stride + 1
            out_w = (w - self.patch_size) // self.stride + 1
            
            # Valid pixels start in (0, 0)
            start_y, start_x = 0, 0
            
        # 2. Build an (out_h, out_w) grid of (S, S, C) patches with step `stride` and flatten into N patches
        windows = sliding_window_view(array, (S, S), axis=(0, 1))[::stride, ::stride] # (out_h, out_w, C, S, S) (or +1 before trim)
        windows = windows[:out_h, :out_w].transpose(0, 1, 3, 4, 2) # (out_h, out_w, S, S, C)
        patches = windows.reshape(-1, S, S, c) # (N, S, S, C)
        
        if not get_coords: 
            return patches # Return only patches if get_coords == False
        
        # 3. Generate Coordinates (Bounding Boxes in pixels)
        # Format: [ymin, xmin, ymax, xmax] relative to original image
        ys = start_y + stride * np.arange(out_h)
        xs = start_x + stride * np.arange(out_w)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        ymin, xmin = yy.ravel(), xx.ravel()
        coords = np.stack([ymin, xmin, ymin + S, xmin + S], axis=1)
                
        return patches, coords    
    
# %%

if __name__ == "__main__":
    # Small 8x8 image with 1 channel
    img = np.arange(8 * 8 * 1).reshape(8, 8, 1)

    # Extract 4x4 patches with 50% overlap (stride = 2) and symmetric padding
    extractor = PatchExtractor(patch_size=4, overlap_percent=0.5, padding=True)
    patches, coords = extractor(img, get_coords=True)

    print("Patches shape:", patches.shape)  # (16, 4, 4, 1)
    print("Coords shape:", coords.shape)    # (16, 4)
    print("First patch:\n", patches[0, :, :, 0])
    print("First coord:", coords[0])      # [-1 -1  3  3]
    print("Last coord:", coords[-1])        # [ 5  5  9  9]

    # Visualize the padded 10x10 grid (top/bottom/left/right padded by 1 px)
    padded_view = np.pad(img[:, :, 0], ((1, 1), (1, 1)), mode="symmetric")
    print("\nPadded 10x10 grid (1px border around original 8x8):\n", padded_view)
    