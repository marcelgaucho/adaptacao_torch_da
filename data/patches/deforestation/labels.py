# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:36:39 2026

@author: Marcel
"""

# %% Imports

# From other libraries
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt; plt.ioff()

# %%

class ChangeLabelBuilder:
    """0 = background, 1 = new deforestation (T2 without T1), ignore_index = already present in T1 deforestation."""
    def __init__(self, target_class: int = 1, ignore_index: int = 255, dil_px=2, ero_px=0):
        self.target_class = target_class
        self.ignore_index = ignore_index
        self.dil_px = dil_px
        self.ero_px = ero_px

    def __call__(self, lbl_t1: np.ndarray, lbl_t2: np.ndarray) -> np.ndarray:
        """Handle the 'ignore' logic for T2. 
        Dilated deforestation in T1 is ignored, plus a ring in T2."""
        # Structuring elements
        d_foot, e_foot = morphology.disk(self.dil_px), morphology.disk(self.ero_px)
        
        # T1 Dilation
        t1_dilated = morphology.dilation(lbl_t1.squeeze(axis=0), footprint=d_foot)
        
        # T2 Rings logic
        t2_base = lbl_t2.squeeze(axis=0)
        t2_dilated = morphology.dilation(t2_base, footprint=d_foot)
        t2_eroded = morphology.erosion(t2_base, footprint=e_foot)
        t2_ignore_ring = t2_dilated - t2_eroded # T2 'ignore' ring
        
        # Mark ignore_index in T2
        new_t2 = t2_eroded.copy().astype(np.uint8)
        new_t2[t2_ignore_ring == self.target_class] = self.ignore_index
        new_t2[t1_dilated == self.target_class] = self.ignore_index
        
        return new_t2[..., np.newaxis] # (H, W, 1)
        
    
# %%

if __name__ == "__main__":
    lbl_t1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    lbl_t2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    builder = ChangeLabelBuilder(target_class=1, ignore_index=255, dil_px=1, ero_px=0)
    saida = builder(lbl_t1[np.newaxis, ...], lbl_t2[np.newaxis, ...])

    print("lbl_t1:\n", lbl_t1)
    print("\nlbl_t2:\n", lbl_t2)
    print("\nsaida:\n", saida[:, :, 0]) # (H, W, 1) -> (H, W) -> print

# %%
