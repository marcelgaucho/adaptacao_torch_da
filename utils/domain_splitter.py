# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:47:54 2026

@author: Marcel
"""

# %% Imports

# Standard libraries
from pathlib import Path
import json
import random

# Other imports
import numpy as np
from PIL import Image
import pandas as pd

# %%

class CHN10DomainManager:
    # Centralized dictionary mapping full city names to their dataset acronyms
    CITY_MAP = {
        "macau": "am",
        "beijing": "bj",
        "dalian": "dl",
        "guangzhou": "gz",
        "hongkong": "hk",
        "hangzhou": "hz",
        "qingdao": "qd",
        "shanghai": "sh",
        "shenzhen": "sz",
        "wuhan": "wh"
    }
    
    def __init__(self, root_dir: str, output_dir: str = "./domain_config"):
        """
        Args:
            root_dir (str or Path): Path to the root directory of the CHN10-CUG dataset.
            output_dir (str or Path): Directory where the split .json file will be saved.
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inverse mapping for easy identification (code -> name)
        self.inverse_city_map = {v: k for k, v in self.CITY_MAP.items()}
        
    def _get_acronyms(self, cities: list) -> list:
        """Converts city names or acronyms into a standardized list of acronyms."""
        acronyms = []
        for city in cities:
            city_lower = city.lower().strip()
            if city_lower in self.CITY_MAP:
                acronyms.append(self.CITY_MAP[city_lower])
            elif city_lower in self.CITY_MAP.values():
                acronyms.append(city_lower)
            else:
                raise ValueError(f"City '{city}' is not recognized in the CHN10-CUG dataset.")
        return acronyms
    
    def _get_gt_path_from_image(self, img_path: str) -> Path:
        """
        Safely maps a training image path (e.g., .../images/am100001_sat.jpg) 
        to its corresponding Ground Truth mask path (e.g., .../gt/am100001_mask.png).
        Raises FileNotFoundError if the mask does not exist.
        """
        img_p = Path(img_path)
        parts = list(img_p.parts)
        
        # Replace 'images' directory with 'gt'
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "gt"
        else:
            raise ValueError(f"Path does not contain an 'images' directory: {img_p}")
        
        gt_dir = Path(*parts[:-1])
        filename = img_p.stem
        
        # Handle suffix differences (e.g., '_sat' -> '_mask')
        if filename.endswith("_sat"):
            base_name = filename[:-4]
        else:
            raise ValueError(f"Image {img_p} doesn't end with '_sat', like other images.")
            
        gt_filename = f"{base_name}_mask.png"
        gt_path = gt_dir / gt_filename
        
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground Truth mask not found for image {img_p}. Expected at: {gt_path}")
            
        return gt_path
    
    def _compute_road_ratio(self, gt_path: Path) -> float:
        """Computes the proportion of road pixels directly from a Ground Truth mask path."""
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground Truth mask not found at: {gt_path}")
        
        mask = Image.open(gt_path).convert('L')
        mask_arr = np.array(mask) > 0
        return np.sum(mask_arr) / mask_arr.size
    
    def find_target_city(self, road_threshold: float = 0.005) -> tuple:
        """
        Identifies the target city based on the number of valid images 
        that meet the minimum road pixel threshold.
        """
        # Get all ground truth paths
        gt_paths = []
        for split in ['train', 'val', 'test']:
            split_dir = self.root_dir / split / 'gt'
            if split_dir.exists():
                gt_paths.extend(list(split_dir.glob('*.png')))
                
        if not gt_paths:
            raise ValueError(f"No masks found in {self.root_dir}. Check the dataset path.")

        # Update total and valid image stats            
        city_stats = {} # this will store the info initially
        
        for gt_path in gt_paths:
            filename = gt_path.stem
            city_code = filename[:2]
            city_name = self.inverse_city_map.get(city_code, city_code).capitalize()
            
            road_ratio = self._compute_road_ratio(gt_path)
            
            if city_name not in city_stats:
                city_stats[city_name] = {'total_images': 0, 'valid_images': 0}
            
            city_stats[city_name]['total_images'] += 1
            if road_ratio >= road_threshold:
                city_stats[city_name]['valid_images'] += 1
        
        # Generate summary dataframe
        data_list = [
            {'City': city, 'Valid_Images_Count': stats['valid_images'], 'Total_Images': stats['total_images']}
            for city, stats in city_stats.items()
        ]
        
        df = pd.DataFrame(data_list).sort_values(by='Valid_Images_Count', ascending=False).reset_index(drop=True)
        
        # Split source cities from target city
        target_city = df.iloc[0]['City']
        source_cities = df[df['City'] != target_city]['City'].tolist()
        
        # Save the domain city configuration file
        config_data = {
            "target_domain": [target_city],
            "source_domain": source_cities,
            "threshold_used": road_threshold
        }
        split_file = self.output_dir / "cities_split.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
            
        print(f"[*] City allocation configuration saved to '{split_file}'")
        return target_city, source_cities, df
    
    def prepare_splits_json(self, domains_dict: dict, ratios=(0.7, 0.15, 0.15), road_threshold=0.005, 
                            seed=42, overwrite=True) -> dict:
        """
        Generates train, validation, and test splits. 
        Applies the road filter ONLY to the source domain training split.
        """
        assert sum(ratios) == 1.0, "The sum of ratios must be exactly 1.0"
                
        output_json_path = self.output_dir / "dataset_splits.json" # output JSON file
        
        if output_json_path.exists() and not overwrite:
            print(f"[*] Split file already exists at {output_json_path}. Loading existing splits...")
            with open(output_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # Generate splits
        print("[*] Generating new splits and applying filtering ONLY to the train set in the source domain...")
        all_images = list(self.root_dir.glob("**/images/*.jpg")) # all images paths
        master_splits_data = {} # this will store each split with their images
        
        # Loop through the domains dict config
        for domain_name, cities in domains_dict.items():
            acronyms = self._get_acronyms(cities)

            # Filter images belonging to the cities in this domain
            domain_images = [
                img.as_posix() for img in all_images 
                if any(img.name.lower().startswith(acronym) for acronym in acronyms)
            ]
            
            # Sort, define seed and shuffle for reproducibility
            domain_images.sort()
            random.seed(seed)
            random.shuffle(domain_images)

            # Perform the base partitioning (train/val/test)
            total = len(domain_images)
            idx_train = int(total * ratios[0])
            idx_val = idx_train + int(total * ratios[1])

            train_imgs = domain_images[:idx_train]
            val_imgs = domain_images[idx_train:idx_val]
            test_imgs = domain_images[idx_val:]

            # CONDITIONAL FILTERING: Only applied to the Training split in the Source domain
            is_source = domain_name == "source_domain"
            if is_source:
                initial_train_count = len(train_imgs)
                filtered_train = []
                for img_path in train_imgs:
                    # Map image path to GT mask path and compute road ratio
                    gt_path = self._get_gt_path_from_image(img_path)
                    if self._compute_road_ratio(gt_path) >= road_threshold:
                        filtered_train.append(img_path)
                train_imgs = filtered_train
                print(f"    [Source Train Filter] Removed {initial_train_count - len(train_imgs)} empty/sparse images. Remaining: {len(train_imgs)}")

            master_splits_data[domain_name] = {
                "train": train_imgs,
                "val": val_imgs,
                "test": test_imgs
            }

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(master_splits_data, f, indent=4)

        print(f"[*] Splits successfully saved to {output_json_path}")
        return master_splits_data
    
# %% Main Execution Block

if __name__ == '__main__':
    DATASET_ROOT = "../CHN10-CUG"
    THRESHOLD = 0.005  # 0.5% road pixels threshold
    
    # 1. Instantiate the domain manager
    manager = CHN10DomainManager(root_dir=DATASET_ROOT, output_dir="./domain_config")
    
    # 2. Automatically discover the best target city and source cities
    print("Analyzing dataset to define the target domain...")
    target_city, source_cities, df_summary = manager.find_target_city(road_threshold=THRESHOLD)
    
    print("\n--- City Viability Report ---")
    print(df_summary.to_string(index=False))
    
    # 3. Prepare the domain dictionary to generate split paths
    config_dict = {
        "source_domain": source_cities,
        "target_domain": [target_city]
    }
    
    # 4. Create splits applying strict filtering to the source training set only
    splits = manager.prepare_splits_json(
        domains_dict=config_dict, 
        ratios=(0.7, 0.15, 0.15), 
        road_threshold=THRESHOLD, 
        overwrite=True
    )
    
    # 5. Clean and comprehensive final summary print
    print("\n" + "=" * 50)
    print(" 📊 DOMAIN ADAPTATION DATASET SUMMARY")
    print("=" * 50)
    
    print(f"\n🎯 Target Domain: {target_city}")
    print(f"   - Train (Unlabeled/Adaptation): {len(splits['target_domain']['train'])} images")
    print(f"   - Val (Original):               {len(splits['target_domain']['val'])} images")
    print(f"   - Test (Original / Unfiltered):   {len(splits['target_domain']['test'])} images")
    
    print(f"\n📦 Source Domains ({len(source_cities)} cities): {', '.join(source_cities)}")
    print(f"   - Train (Filtered >= {THRESHOLD*100}%):    {len(splits['source_domain']['train'])} images")
    print(f"   - Val (Original):               {len(splits['source_domain']['val'])} images")
    print(f"   - Test (Original):              {len(splits['source_domain']['test'])} images")
    
    print("=" * 50)




        