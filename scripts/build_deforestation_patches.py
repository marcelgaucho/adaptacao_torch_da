#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:49:53 2026

@author: rotunno
"""

# %% Imports

# From our modules
from data.patches.deforestation import (DatasetLayout, RasterReader, ChangeLabelBuilder, PatchExtractor,
                                        FusedPatchPipeline, PatchStore)

# From other modules
import numpy as np

# %%

def main():
    # Extraction parameters
    patch_size = 128
    overlap = 0.9
    padding = True
    out_dir = 'data/deforest_patches'
    
    # Dataset parameters
    root = r'../deforestation_dataset/'
    target_class = 1
    domains = ["MA", "PA", "RO"]
    image_dtype = np.uint16
        
    # Label parameters
    ignore_class = 255
    dilation_px = 2 
    erosion_px = 0
    
    layout = DatasetLayout(root) # dataset structure
    extractor = PatchExtractor(patch_size, overlap, padding=padding) # patch extractor
    store = PatchStore(out_dir, img_dtype=image_dtype) # manage patch and metadata saving

    for domain in domains: # loop though domains in dataset
        for split in ["train", "valid", "test"]: # loop through the data splits
            pipeline = FusedPatchPipeline(
                RasterReader(), 
                ChangeLabelBuilder(target_class=target_class,
                                   ignore_index=ignore_class, 
                                   dil_px=dilation_px, 
                                   ero_px=erosion_px), 
                extractor) # patch extraction pipeline
            n = store.write(pipeline, layout.samples(domain, split), domain, split) # store the patches
            print(f"{domain}/{split}: {n} patches")

# %%

if __name__ == "__main__":
    main()    
