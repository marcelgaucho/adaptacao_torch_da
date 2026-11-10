# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:42:02 2026

@author: Marcel
"""

# %% Import Libraries

from pathlib import Path
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.dataset import CHN10CUGDataset

# %%

# Define root data directory using pathlib
ROOT_DATA_DIR = Path("./path/to/CHN10-CUG")

# Define augmentation pipeline with Albumentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.75),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# %%

# Create the Dataset
train_dataset = CHN10CUGDataset(
    root_dir=ROOT_DATA_DIR, 
    split='train', 
    transform=train_transform
)

# Create the DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)