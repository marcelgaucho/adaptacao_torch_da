# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:03:04 2026

@author: Marcel
"""

# %% Import Libraries

import glob
import random
from pathlib import Path

# %%

class CHN10DomainManager:
    # Centralized dictionary mapping full city names to their dataset acronyms
    CITY_MAP = {
        "macau": "am",       # Aomen / Macau
        "beijing": "bj",     # Beijing
        "dalian": "dl",      # Dalian
        "guangzhou": "gz",   # Guangzhou
        "hongkong": "hk",    # Hong Kong
        "hangzhou": "hz",    # Hangzhou
        "qingdao": "qd",     # Qingdao
        "shanghai": "sh",    # Shanghai
        "shenzhen": "sz",    # Shenzhen
        "wuhan": "wh"        # Wuhan
    }

    def __init__(self, root_dir, output_txt_dir="./data_splits"):
        """
        Args:
            root_dir (str or Path): Path to the root directory of the CHN10-CUG dataset.
            output_txt_dir (str or Path): Directory where the split .txt files will be saved.
        """
        self.root_dir = Path(root_dir)
        self.output_txt_dir = Path(output_txt_dir)
        
        # Create the output directory for txt splits if it does not exist
        self.output_txt_dir.mkdir(parents=True, exist_ok=True)

    def prepare_split(self, domain_name, cities, ratios=(0.7, 0.15, 0.15), seed=42):
        """
        Dynamically creates Train, Validation, and Test .txt files for a given domain using pathlib.
        ratios: Tuple containing (train_ratio, val_ratio, test_ratio) summing up to 1.0.
        """
        assert sum(ratios) == 1.0, "The sum of ratios must be exactly 1.0"

        # Define destination paths for the 3 metadata files
        train_txt = self.output_txt_dir / f"{domain_name}_train.txt"
        val_txt   = self.output_txt_dir / f"{domain_name}_val.txt"
        test_txt  = self.output_txt_dir / f"{domain_name}_test.txt"

        # If all three split files already exist, skip recreation to preserve data consistency
        if train_txt.exists() and val_txt.exists() and test_txt.exists():
            return train_txt, val_txt, test_txt

        print(f"[*] Generating splits (Train/Val/Test) for domain: {domain_name}...")
        
        # Convert user-requested city names to their corresponding dataset acronyms
        target_acronyms = []
        for city in cities:
            city_lower = city.lower().strip()
            if city_lower in self.CITY_MAP:
                target_acronyms.append(self.CITY_MAP[city_lower])
            else:
                # Fallback in case the user already provided the acronym (e.g., "am" instead of "macau")
                if city_lower in self.CITY_MAP.values():
                    target_acronyms.append(city_lower)
                else:
                    raise ValueError(f"City '{city}' is not recognized in CHN10-CUG dataset.")

        # Collect all image paths. By targeting specific subfolders, we inherently exclude the 'Graph' folder.
        # This scans across train/val/test folders but only looks inside the 'images' subdirectories.
        all_images = list(self.root_dir.glob("**/images/*.png"))
        
        # Filter images based on the mapped city acronym prefixes (e.g., checking if filename starts with 'am')
        filtered = [
            img for img in all_images 
            if any(img.name.lower().startswith(acronym) for acronym in target_acronyms)
        ]
        
        # Stable sort to ensure reproducibility across different operating systems before shuffling
        filtered.sort()
        
        # Shuffle dataset with a fixed seed to prevent data leakage in future runs
        random.seed(seed)
        random.shuffle(filtered)

        # Compute exact index slicing points based on user ratios
        total_samples = len(filtered)
        idx_train_end = int(total_samples * ratios[0])
        idx_val_end   = idx_train_end + int(total_samples * ratios[1])

        train_list = filtered[:idx_train_end]
        val_list   = filtered[idx_train_end:idx_val_end]
        test_list  = filtered[idx_val_end:]

        # Save the 3 splits as persistent files, writing paths in POSIX format (forward slashes)
        for lst, txt_path in zip([train_list, val_list, test_list], [train_txt, val_txt, test_txt]):
            with txt_path.open("w", encoding="utf-8") as f:
                for item in lst: 
                    f.write(f"{item.as_posix()}\n")

        return train_txt, val_txt, test_txt

    
# %% Test script 

if __name__ == '__main__':
    manager = CHN10DomainManager(root_dir="./CHN10-CUG")

    # Creating splits files
    src_train, src_val, src_test = manager.prepare_split("source", cities=["beijing", "shanghai"])
    tgt_train, tgt_val, tgt_test = manager.prepare_split("target", cities=["hongkong"])

        
        
        