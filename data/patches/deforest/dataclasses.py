#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:52:14 2026

@author: rotunno
"""

# %% Imports

# From standard modules
from dataclasses import dataclass
from pathlib import Path

# %%

@dataclass
class PatchConfig:
    patch_size: int = 64
    overlap: float = 0.5
    padding: bool = True
    nodata_value: float = 0.0
    nodata_tolerance: float = 0.0
    target_class: int = 1
    ignore_class: int = 255
    min_object_ratio: float = 0.02
    max_nodata_ratio: float = 0.5
    preprocessed_dir: Path | None = None
    
    # Para avaliar se retiro
    ignore_t1_in_label: bool = True
    filter_by_object_ratio: bool = True

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than zero.")            
            
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be within [0.0, 1.0).")
        
        if not (0.0 <= self.min_object_ratio <= 1.0):
            raise ValueError("min_object_ratio must be within [0.0, 1.0].")
            
        if not (0.0 <= self.max_nodata_ratio <= 1.0):
            raise ValueError("max_nodata_ratio must be within [0.0, 1.0].")
            
        if self.preprocessed_dir is not None:
            self.preprocessed_dir = Path(self.preprocessed_dir)
            
    @property
    def overlap_px(self) -> int:
        return int(self.patch_size * self.overlap)
    
    @property
    def stride(self) -> int:
        stride = self.patch_size - self.overlap_px
        if stride < 1:
            raise ValueError("Stride must be >= 1. Check patch_size and overlap.")
        return stride
            
# %%

@dataclass(frozen=True)
class SamplePair:
    domain: str
    split: str
    tile_id: str
    img_t1_path: Path
    img_t2_path: Path
    lbl_t1_path: Path
    lbl_t2_path: Path
