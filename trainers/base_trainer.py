#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:58:12 2026

@author: rotunno
"""

# %% Imports

# Standard imports
from itertools import cycle
from abc import ABC, abstractmethod
from pathlib import Path

# Torch imports
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# From own modules
from evaluators.segmentation_evaluator import SegmentationEvaluator

# %%

class BaseTrainer(ABC):
    """
    Base trainer class handling common setup for semantic segmentation domain adaption experiments.    
    """
    def __init__(self, model, train_source_loader, train_target_loader, 
                 val_source_loader, val_target_loader, log_dir="runs/experiment",
                 save_dir="checkpoints", device="cuda", num_classes=11):
        self.model = model.to(device) # model
        
        # All Loaders used by trainer classes
        self.train_source_loader = train_source_loader
        self.train_target_loader = train_target_loader
        self.val_source_loader = val_source_loader
        self.val_target_loader = val_target_loader
        
        # General config
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(exist_ok=True, parents=True) # Create checkpoints dir
        
        # Evaluator & TensorBoard Writer
        self.evaluator = SegmentationEvaluator(num_classes=num_classes, device=device)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: dict, epoch: int, prefix: str = "val"):
        """Helper method to log scalar metrics to TensorBoard cleanly."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, epoch)

    def close(self):
        """Closes the TensorBoard writer safely."""
        self.writer.close()
        
    def _save_checkpoint(self, filename: str):
        """Helper method to save the model checkpoint."""
        filepath = Path(self.save_dir) / filename
        torch.save(self.model.state_dict(), filepath)
        print(f"[*] Checkpoint saved in: {filepath}")
        
    @abstractmethod
    def train(self, num_epochs: int):
        """
        Subclasses must implement the training loop with their domain adaptation strategy.
        """
        pass

