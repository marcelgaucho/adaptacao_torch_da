# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:49:09 2026

@author: Marcel
"""

# %% Imports

# From standard modules
from dataclasses import dataclass
from pathlib import Path

# %% Dataset element

@dataclass(frozen=True)
class TileSample:
    domain: str
    split: str
    name: str
    img_t1: Path
    img_t2: Path
    lbl_t1: Path
    lbl_t2: Path
    
# %% Dataset structure

class DatasetLayout:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        
    def samples(self, domain: str, split: str) -> list[TileSample]:
        base = self.root / domain / split
        out = []
        
        # Only consider the "*.tif" files
        for p in sorted((base / "image" / "t1").glob("*.tif")):
            s = TileSample(
                domain, split, p.stem,
                img_t1=p,
                img_t2=base / "image" / "t2" / p.name,
                lbl_t1=base / "label" / "t1" / p.name,
                lbl_t2=base / "label" / "t2" / p.name,
            )
            
            missing = [f for f in (s.img_t2, s.lbl_t1, s.lbl_t2) if not f.exists()] # search for missing files
            if missing:
                raise FileNotFoundError(f"{s.name}: faltando {missing}")
                
            out.append(s)
        
        return out

# %%

if __name__ == '__main__':
    # Dataset info
    root = '../deforestation_dataset/'
    domain = 'PA'
    split = 'train'
    
    layout = DatasetLayout(root)
    tiles = layout.samples(domain, split) # tile list
    
    for tile in tiles: print(tile, end='\n\n') # skip one line to print