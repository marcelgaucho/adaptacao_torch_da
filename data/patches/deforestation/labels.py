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
    """Generates change labels for T2 based on T1 (past deforestation) and T2 (new deforestation). 
    Ignores a buffer over T1 and a ring around T2. Expected input and output format: (H, W, 1).

    Classes:
        (background): 0 is the common choice
        (target_class): new deforestation in T2
        ignore_index: value (in uint8) for pixels excluded from loss calculation
    """
    def __init__(self, target_class: int = 1, ignore_index: int = 255, dil_px=2, ero_px=0):
        self.target_class = target_class
        self.ignore_index = ignore_index
        self.dil_px = dil_px
        self.ero_px = ero_px
        
        # Structuring elements
        self._d_foot, self._e_foot = morphology.disk(self.dil_px), morphology.disk(self.ero_px)
        
    @staticmethod
    def _to_hw(arr: np.ndarray, name: str) -> np.ndarray:
        # (H, W, 1) -> (H, W)
        if arr.ndim != 3 or arr.shape[-1] != 1:
            raise ValueError(f"{name} must have shape (H, W, 1), got {arr.shape}.")
        return arr[..., 0]

    def __call__(self, lbl_t1: np.ndarray, lbl_t2: np.ndarray) -> np.ndarray:
        """Handle the 'ignore' logic for T2. 
        Dilated deforestation in T1 is ignored, plus a ring in T2."""
        t1_2d = self._to_hw(lbl_t1, "lbl_t1")
        t2_2d = self._to_hw(lbl_t2, "lbl_t2")
        
        if t1_2d.shape != t2_2d.shape:
            raise ValueError(f"Shape mismatch: lbl_t1 {t1_2d.shape} vs lbl_t2 {t2_2d.shape}.")
 
        # T1 Dilation. Buffer over T1 is ignored in T2
        t1_dilated = morphology.dilation(t1_2d == self.target_class, footprint=self._d_foot)
        
        # T2 Rings logic. A ring around T2 (space between dilation and erosion) is also ignored
        t2_base = t2_2d == self.target_class
        t2_dilated = morphology.dilation(t2_base, footprint=self._d_foot)
        t2_eroded = morphology.erosion(t2_base, footprint=self._e_foot)
        t2_ignore_ring = t2_dilated & (~t2_eroded) # T2 'ignore' ring
        
        # Mark ignore_index in T2
        new_t2 = t2_2d.astype(np.uint8, copy=True)
        new_t2[t1_dilated | t2_ignore_ring] = self.ignore_index
        
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
    saida = builder(lbl_t1[..., np.newaxis], lbl_t2[..., np.newaxis])

    print("lbl_t1:\n", lbl_t1)
    print("\nlbl_t2:\n", lbl_t2)
    print("\nsaida:\n", saida[:, :, 0]) # (H, W, 1) -> (H, W) -> print
