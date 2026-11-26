#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:49:53 2026

@author: rotunno
"""

# %% Imports

# From our modules
from data.patches.deforestation import (DatasetLayout, RasterReader, ChangeLabelBuilder, PatchExtractor,
                                        FusedPatchPipeline, PatchStore, IGNORE_INDEX)

# %%

def main():
    # Extraction parameters
    patch_size = 128
    overlap = 0.9
    padding = True
    out_dir = 'data/deforest_patches'
    
    # Dataset parameters
    root = r'../../deforestation_dataset/'
    target_class = 1
        
    # Output label parameters
    ignore_class = 255
    
    
