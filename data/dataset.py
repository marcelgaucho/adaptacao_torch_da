# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:06:56 2026

@author: Marcel
"""

# %% Import Libraries

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%

class CHN10CUGDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str or Path): Root directory where the CHN10-CUG dataset is stored.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Albumentations data augmentation pipeline.
        """
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.image_dir = self.split_dir / 'images'
        self.mask_dir = self.split_dir / 'gt'
        
        # List and sort files safely using pathlib (filtering files to avoid accidental subdirectories)
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.is_file()])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # The corresponding mask path shares the same filename as the image
        mask_path = self.mask_dir / img_path.name
        
        # 1. Read the image (Convert Path to str, as OpenCV prefers strings)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Read the mask (grayscale)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 3. Binarize the road mask (ensure binary values 0 and 1)
        mask = (mask > 127).astype(np.uint8)

        # 4. Apply data augmentations (D-8 or others)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure the mask is float for loss functions like BCEWithLogitsLoss
        if isinstance(mask, torch.Tensor):
            mask = mask.float()

        return image, mask