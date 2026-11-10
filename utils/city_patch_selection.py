# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:02:11 2026

@author: Marcel
"""

# %% Imports

# Standard Imports
import json
from pathlib import Path

# Other imports
import numpy as np
import pandas as pd
from PIL import Image

# %% Find city function

def find_target_city(dataset_root: str, threshold: float = 0.01) -> tuple:
    """
    Identifies the city with the greatest number of images with at least
    the specified threshold percentage of road pixels.    
    """
    gt_paths = []
    # Iterate over split folders to list the ground truth image paths 
    for split in ['train', 'val', 'test']:
        split_dir = Path(dataset_root) / split / 'gt'
        if split_dir.exists():
            gt_paths.extend( list(split_dir.glob('*.png')) )
            
    if not gt_paths:
        raise ValueError(f"No masks found in {dataset_root}. Check the path.") # paths list is empty
        
    city_stats = {}
    
    # Dictionary mapping of code to city name
    city_mapping = {
        "am": "Macau",       # Aomen / Macau
        "bj": "Beijing",     # Beijing
        "dl": "Dalian",      # Dalian
        "gz": "Guangzhou",   # Guangzhou
        "hk": "Hongkong",    # Hong Kong
        "hz": "Hangzhou",    # Hangzhou
        "qd": "Qingdao",     # Qingdao
        "sh": "Shanghai",    # Shanghai
        "sz": "Shenzhen",    # Shenzhen
        "wh": "Wuhan"        # Wuhan
    }
    
    # Loop through city list images computing road pixel ratios
    for gt_path in gt_paths:
        filename = gt_path.stem # Extract only the filename (without extension)
        
        # Get the city name through the city code 
        city_code = filename[:2]
        city_name = city_mapping.get(city_code, city_code)
        
        # Load image mask in grayscale
        mask = Image.open(gt_path).convert('L')
        mask_arr = np.array(mask) > 0 # True where pixel is road
        
        road_ratio = np.sum(mask_arr) / mask_arr.size # Road proportion in image
        
        if city_name not in city_stats:
            city_stats[city_name] = {'total_images': 0, 'valid_images': 0} # Set initial values for a new city
        
        city_stats[city_name]['total_images'] += 1 # Increment number of analysed images
        
        if road_ratio >= threshold:
            city_stats[city_name]['valid_images'] += 1 # Valid images increases if condition is met
     
    # Convert results to a DataFrame for easier visualization
    data_list = []
    for city, stats in city_stats.items():
        data_list.append({
            'City': city,
            'Valid_Images_Count': stats['valid_images'],
            'Total_Images': stats['total_images']           
        })
        
    df = pd.DataFrame(data_list)
        
    # Sort from highest to lowest number of valid images
    df = df.sort_values(by='Valid_Images_Count', ascending=False).reset_index(drop=True)
    
    # The city with the highest number of valid images is chosen as target
    target_city = df.iloc[0]['City']
    
    # The remaining cities are the source
    source_cities = df[df['City'] != target_city]['City'].tolist()
    
    return target_city, source_cities, df        
    

# %% Execute the function to find city

if __name__ == "__main__":
    # Specify the dataset root path and the threshold
    DATASET_ROOT = "../CHN10-CUG"  
    THRESHOLD = 0.005  # Threshold percentage of the pixels
    
    print("Analyzing the dataset and computing statistics per city...")
    target_city, source_cities, df_summary = find_target_city(DATASET_ROOT, threshold=THRESHOLD)
    
    print(f"\n--- Report of Images with >= {THRESHOLD*100}% Road ---")
    print(df_summary.to_string(index=False))
    
    print("\n==================================================")
    print(f"🎯 TARGET Domain (held out): {target_city}")
    print(f"📦 SOURCE Domains (9 cities for training): {source_cities}")
    print("==================================================")

    # Save the configuration to JSON for use in the training script
    config_data = {
        "target_domain": [target_city],
        "source_domain": source_cities,
        "threshold_used": THRESHOLD
    }

    split_file = Path("domain_config/cities_split.json") # File path to JSON
    split_file.parent.mkdir(exist_ok=True, parents=True) # Create the file folder
    with open(split_file, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"\nConfiguration saved successfully in '{split_file}'.")


