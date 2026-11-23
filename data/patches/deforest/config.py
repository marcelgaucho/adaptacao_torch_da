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
    min_object_ratio: float = 0.02
    preprocessed_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than zero.")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must be within [0.0, 1.0).")
        if self.preprocessed_dir is not None:
            self.preprocessed_dir = Path(self.preprocessed_dir)