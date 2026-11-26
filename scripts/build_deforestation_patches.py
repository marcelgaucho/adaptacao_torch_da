#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:49:53 2026

@author: rotunno
"""

# %% Imports

# Pasta scripts/build_deforestation_patches.py

# From our modules
from data.patches.deforestation import (DatasetLayout, RasterReader, ChangeLabelBuilder, PatchExtractor,
                                        FusedPatchPipeline, PatchStore, IGNORE_INDEX)
