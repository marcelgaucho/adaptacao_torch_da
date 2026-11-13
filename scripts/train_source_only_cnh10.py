#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:34:08 2026

@author: rotunno
"""

# %% Imports

# From Pytorch
import torch
from torch.utils.data import DataLoader

# From own modules
from models.resunet import ResUnet
from data.chn10_json_dataset import CHN10SegmentationDataset
from trainers.source_only import SourceOnlyTrainer

# %% Main function

def main():
    # 1. Experiment configurations
    json_path = "domain_config/dataset_splits.json"
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    num_classes = 2  # Binary segmentation (background + target)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[*] Starting Source-Only Training on device: {device}")
    
    # 2. Load Datasets for Source Domain (Training & Validation) and Target Domain (Only Validation)
    train_source_ds = CHN10SegmentationDataset(json_path, domain_key="source_domain", split="train")
    val_source_ds = CHN10SegmentationDataset(json_path, domain_key="source_domain", split="val")
    val_target_ds = CHN10SegmentationDataset(json_path, domain_key="target_domain", split="val")
    
    # 3. Create DataLoaders
    train_source_loader = DataLoader(train_source_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_source_loader = DataLoader(val_source_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_target_loader = DataLoader(val_target_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 4. Initialize ResUnet Model
    model = ResUnet(in_channels=3, out_channels=num_classes, features=[64, 128, 256, 512])
    
    # 5. Initialize the SourceOnlyTrainer
    trainer = SourceOnlyTrainer(
        model=model,
        train_source_loader=train_source_loader,
        train_target_loader=None,  # Not used in pure source-only training
        val_source_loader=val_source_loader,
        val_target_loader=val_target_loader,
        lr=learning_rate,
        num_classes=num_classes,
        device=device,
        log_dir="runs/source_only_experiment",
        save_dir="checkpoints"
    )
    
    # 6. Execute Training Loop
    trainer.train(num_epochs=num_epochs)
    print("[*] Source-Only training completed successfully!")
    
# %% Execute main

if __name__ == "__main__":
    main()