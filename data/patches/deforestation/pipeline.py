#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:39:28 2026

@author: rotunno
"""

# %% Imports

# From standard modules
from dataclasses import dataclass

# From other modules
import numpy as np

# %% Class to store extracted patches for a tile

@dataclass
class TilePatches:
    x: np.ndarray        # (N, S, S, C_t1 + C_t2)  early fusion
    y: np.ndarray        # (N, S, S)
    coords: np.ndarray   # (N, 4) [ymin, xmin, ymax, xmax]
    tile_hw: tuple       # (H, W) of original tile
    
# %% Class Pipeline

class FusedPatchPipeline:
    """Reads a TileSample, fuses T1+T2 along the channel axis, builds the target, and extracts the patches."""
    
    def __init__(self, reader, label_builder, extractor, ignore_index=255):
        self.reader = reader # Read a raster
        self.label_builder = label_builder # Build label with ignored pixels
        self.extractor = extractor # Extract the patches
        self.ignore_index = ignore_index # Ignore index
    
    def extract(self, sample) -> TilePatches:
        img_t1, img_t2 = self.reader.read(sample.img_t1), self.reader.read(sample.img_t2) # Read images
        lbl_t1, lbl_t2 = self.reader.read(sample.lbl_t1), self.reader.read(sample.lbl_t2) # Read labels
        
        shapes = {a.shape[:2] for a in (img_t1, img_t2, lbl_t1, lbl_t2)} # (H, W) for all tiles
        if len(shapes) != 1: # True if there is shape mismatch for a tile
            raise ValueError(f"{sample.name}: mismatched shapes across T1/T2/labels: {shapes}")
            
        fused = np.concatenate([img_t1, img_t2], axis=-1) # image fusion
        target = self.label_builder(lbl_t1, lbl_t2) # build label
        
        x, coords = self.extractor(fused) # X patch extraction
        y, coords_y = self.extractor(target) # Y patch extraction
        
        if not np.array_equal(coords, coords_y): # x and y patch coordinates must be equal
            raise ValueError(f"{sample.name}: x and y extracted patch coordinates do not match")
        
        return TilePatches(x, y.squeeze(axis=-1), coords, fused.shape[:2]) # Return an instance of TilePatches
