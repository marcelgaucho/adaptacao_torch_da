#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:27:59 2026

@author: rotunno
"""

# %% Import libraries

# From standard libraries
from dataclasses import dataclass
from pathlib import Path
import json

# %%

@dataclass(frozen=True)
class MinMaxStats:
    """Minimum and maximum per channel (order: T1 bands, followed by T2 bands)."""


@dataclass(frozen=True)
class MinMaxStats:
    """Mínimo e máximo por canal (ordem: bandas de T1 e depois as de T2)."""
    min: tuple
    max: tuple

    @staticmethod
    def path_for(root, domain) -> Path:
        return Path(root) / domain / "stats.json"

    def save(self, path):
        Path(path).write_text(json.dumps({"min": self.min, "max": self.max}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(tuple(d["min"]), tuple(d["max"]))






def minmax_normalize(data, min_value=None, max_value=None):
    """Min-max normalization to [0, 1] for concatenated patches. 
       Use input min and max values if passed."""
    # 1. Force math to be float32 for accuracy
    data = data.astype(np.float32)
    if min_value is None:
        # Min/Max per channel across the whole dataset
        min_value = np.min(data, axis=(0, 1, 2))
        max_value = np.max(data, axis=(0, 1, 2))
        
    # 2. Perform the math
    normalized = (data - min_value) / (max_value - min_value + np.float32(1e-7))
    
    # 3. Return normalized and max and min values used in normalization
    return normalized, min_value, max_value




def compute_channel_stats(images_path: Path, out_path: Path, domain: str, split: str) -> dict:
    # mmap_mode="r" ajuda se images.npy for grande
    images = np.load(images_path, mmap_mode="r")

    # Média e std por canal C, agregando sobre N, H, W
    mean = images.mean(axis=(0, 2, 3), dtype=np.float64)
    std = images.std(axis=(0, 2, 3), dtype=np.float64)

    stats = {
        "domain": domain,
        "split": split,
        "num_channels": int(images.shape[1]),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=4))
    return stats