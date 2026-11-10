# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:26:13 2026

@author: Marcel
"""

# %% Imports

import numpy as np
import pandas as pd

# %%

def create_city_groups(
    df: pd.DataFrame, group_a: list[str], group_b: list[str]
) -> pd.DataFrame:
    """Groups cities into two segments and calculates the share of valid images for each group."""
    # Prevent modifying the original DataFrame passed as an argument
    df = df.copy()

    # Define grouping rules based on the provided city lists
    conditions = [df['City'].isin(group_a), df['City'].isin(group_b)]
    group_names = ['Group A', 'Group B']

    df['Group'] = np.select(conditions, group_names, default='Others')

    # Aggregate image counts by group
    df_comparison = (
        df.groupby('Group')[['Valid_Images_Count', 'Total_Images']]
        .sum()
        .reset_index()
    )

    # Calculate each group's share relative to the global total of valid images
    total_valids = df['Valid_Images_Count'].sum()
    df_comparison['%_of_Total_Valids'] = (
        df_comparison['Valid_Images_Count'] / total_valids * 100
    ).round(2)

    return df_comparison

# %% Test group domain division in groups

if __name__ == '__main__':
    # Create DataFrame with road information from dict
    cities_dict = {'City': {0: 'Qingdao', 1: 'Wuhan', 2: 'Shanghai', 3: 'Hangzhou', 4: 'Dalian', 5: 'Beijing', 6: 'Shenzhen', 7: 'Guangzhou', 8: 'Macau', 9: 'Hongkong'}, 'Valid_Images_Count': {0: 825, 1: 673, 2: 642, 3: 574, 4: 544, 5: 510, 6: 488, 7: 455, 8: 409, 9: 143}, 'Total_Images': {0: 992, 1: 864, 2: 891, 3: 756, 4: 837, 5: 575, 6: 961, 7: 924, 8: 950, 9: 270}}
    df_cities = pd.DataFrame(cities_dict)
    
    # Groups division
    group_A_cities = ['Macau', 'Hongkong', 'Shenzhen', 'Guangzhou', 'Shanghai']
    group_B_cities = ['Hangzhou', 'Wuhan', 'Qingdao', 'Beijing', 'Dalian']
    
    # Make comparison and print 
    df_comparison = create_city_groups(df_cities, group_A_cities, group_B_cities)
    print(df_comparison)


