# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:03:04 2026

@author: Marcel
"""

# %% Import Libraries

import json
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

    def __init__(self, root_dir, output_dir="./domain_config"):
        """
        Args:
            root_dir (str or Path): Path to the root directory of the CHN10-CUG dataset.
            output_dir (str or Path): Directory where the split .json file will be saved.
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        
    def prepare_splits_json(self, domains_dict: dict, ratios=(0.7, 0.15, 0.15), seed=42, overwrite=True):
        """
        Prepares train/val/test splits for multiple domains (e.g., source and target) 
        and saves them cleanly into a single structured JSON file.
        
        domains_dict format: {"source": ["beijing", "shanghai"], "target": ["hongkong"]}
        """
        assert sum(ratios) == 1.0, "The sum of ratios must be exactly 1.0"
        
        output_json_path = self.output_dir / "dataset_splits.json" # output json

        # Return early if file exists and overwrite is False
        if output_json_path.exists() and not overwrite:
            print(f"[*] Split file already exists at {output_json_path}. Loading existing splits.")
            with open(output_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print("[*] Generating new dataset splits...")
        all_images = list(self.root_dir.glob("**/images/*.jpg"))
        master_splits_data = {}

        for domain_name, cities in domains_dict.items():
            # Convert city names to acronyms
            target_acronyms = []
            for city in cities:
                city_lower = city.lower().strip()
                if city_lower in self.CITY_MAP:
                    target_acronyms.append(self.CITY_MAP[city_lower])
                elif city_lower in self.CITY_MAP.values():
                    target_acronyms.append(city_lower)
                else:
                    raise ValueError(f"City '{city}' is not recognized in CHN10-CUG dataset.")

            # Filter images for current domain
            filtered = [
                img.as_posix() for img in all_images 
                if any(img.name.lower().startswith(acronym) for acronym in target_acronyms)
            ]
            
            filtered.sort()
            random.seed(seed)
            random.shuffle(filtered)

            # Slicing splits
            total = len(filtered)
            idx_train = int(total * ratios[0])
            idx_val = idx_train + int(total * ratios[1])

            master_splits_data[domain_name] = {
                "train": filtered[:idx_train],
                "val": filtered[idx_train:idx_val],
                "test": filtered[idx_val:]
            }

        # Save everything into a single clean JSON file
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(master_splits_data, f, indent=4)

        print(f"[*] Splits successfully saved to {output_json_path}")
        return master_splits_data
   
# %% Test script 

if __name__ == '__main__':
    split_config_path = Path("domain_config/cities_split.json") # Path to JSON with the cities split
    
    if not split_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {split_config_path}. Run the city finder script first.") # Raise error if JSON path doesn't exists
        
    with open(split_config_path, "r") as f:
        config = json.load(f) # Load saved config
    
    threshold_used = config.pop("threshold_used", None) # Remove key to ensure format
    
    source_cities = config["source_domain"]
    target_cities = config["target_domain"] # Note: JSON salves city as a list, e.g. ["Qingdao"]
    
    print(f"[*] Loaded cities from {split_config_path}")
    print(f"    - Source Cities: {source_cities}")
    print(f"    - Target Cities: {target_cities}")
        
    manager = CHN10DomainManager(root_dir="../CHN10-CUG") # Instanciate domain manager

    # Generate the split files using the input JSON data
    splits = manager.prepare_splits_json(config, overwrite=True)
    
    print(f"Total source train images: {len(splits['source_domain']['train'])}")
    print(f"Total target test images: {len(splits['target_domain']['test'])}")