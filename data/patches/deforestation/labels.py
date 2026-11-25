# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:36:39 2026

@author: Marcel
"""

# %% Imports

# From other libraries
from osgeo import gdal, gdal_array
import numpy as np
from skimage import morphology



# %%

class ChangeLabelBuilder:
    """0 = background, 1 = new deforestation (T2 \ T1), ignore_index = already present in T1 deforestation."""
    def __init__(self, target_class, ignore_index: int = 255):
        self.target_class = target_class
        self.ignore_index = ignore_index

    def __call__(self, lbl_t1: np.ndarray, lbl_t2: np.ndarray,
                 dil_px=2, ero_px=0) -> np.ndarray:
        """Handle the 'ignore' logic for T2. 
        Dilated deforestation in T1 is ignored, plus a ring in T2."""
        # Structuring elements
        d_foot, e_foot = morphology.disk(dil_px), morphology.disk(ero_px)
        
        # T1 Dilation
        t1_dilated = morphology.dilation(lbl_t1.squeeze(), footprint=d_foot)
        
        # T2 Rings logic
        t2_base = lbl_t2.squeeze()
        t2_dilated = morphology.dilation(t2_base, footprint=d_foot)
        t2_eroded = morphology.erosion(t2_base, footprint=e_foot)
        t2_ignore_ring = t2_dilated - t2_eroded # T2 'ignore' ring
        
        # Mark ignore_index in T2
        new_t2 = t2_eroded.copy().astype(np.uint8)
        new_t2[t2_ignore_ring == self.target_class] = 255
        new_t2[t1_dilated == self.target_class] = 255
        
        return new_t2 # or new_t2[..., np.newaxis] (H, W, 1)?
    

        

        









