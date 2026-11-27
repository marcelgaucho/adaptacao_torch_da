#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:39:28 2026

@author: rotunno
"""

# %% Imports

# From standard modules
from dataclasses import dataclass

# From standard for testing purposes
from types import SimpleNamespace
from pathlib import Path

# From our modules
from .extraction import PatchExtractor

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
    
    def __init__(self, reader, label_builder, extractor):
        self.reader = reader # Read a raster
        self.label_builder = label_builder # Build label with ignored pixels
        self.extractor = extractor # Extract the patches
    
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

# %%

if __name__ == "__main__":

    # Synthetic data
    H, W = 8, 8

    img_t1 = np.ones((H, W, 2), dtype=np.float32)
    img_t2 = np.full((H, W, 2), 2, dtype=np.float32)

    lbl_t1 = np.zeros((H, W, 1), dtype=np.uint8)
    lbl_t2 = np.zeros((H, W, 1), dtype=np.uint8)

    lbl_t1[2:6, 2:6] = 1
    lbl_t2[3:5, 3:5] = 1

    # Mock reader
    class MockReader:
        def __init__(self, data):
            self.data = data

        def read(self, path):
            return self.data[Path(path).name]

    reader = MockReader({
        "img_t1.tif": img_t1,
        "img_t2.tif": img_t2,
        "lbl_t1.tif": lbl_t1,
        "lbl_t2.tif": lbl_t2,
    })

    # Label builder
    def label_builder(lbl_t1, lbl_t2):
        return (lbl_t1 != lbl_t2).astype(np.uint8)

    # Patch extractor
    extractor = PatchExtractor(
        patch_size=4,
        overlap_percent=0.25,
        padding=False
    )

    # Sample
    sample = SimpleNamespace(
        name="test_tile",
        img_t1=Path("img_t1.tif"),
        img_t2=Path("img_t2.tif"),
        lbl_t1=Path("lbl_t1.tif"),
        lbl_t2=Path("lbl_t2.tif"),
    )

    # Pipeline
    pipeline = FusedPatchPipeline(reader, label_builder, extractor)
    result = pipeline.extract(sample)

    # Expected outputs
    expected_x, expected_coords = extractor(
        np.concatenate([img_t1, img_t2], axis=-1)
    )

    expected_y, expected_coords_y = extractor(
        label_builder(lbl_t1, lbl_t2)
    )

    # Shape assertions
    assert result.x.shape == expected_x.shape
    assert result.y.shape == expected_y.squeeze(axis=-1).shape
    assert result.coords.shape == expected_coords.shape
    assert result.tile_hw == (H, W)

    # Content assertions
    assert np.array_equal(result.x, expected_x)
    assert np.array_equal(result.y, expected_y.squeeze(axis=-1))

    # Coordinate assertions
    assert np.array_equal(result.coords, expected_coords)
    assert np.array_equal(expected_coords, expected_coords_y)

    # Fusion assertions
    assert np.all(result.x[..., :2] == 1)
    assert np.all(result.x[..., 2:] == 2)

    print("All pipeline tests passed!")
    print(f"Number of patches: {len(result.x)}")
    print(f"Patch shape: {result.x.shape[1:]}")
    print(f"Coordinates:\n{result.coords}")
