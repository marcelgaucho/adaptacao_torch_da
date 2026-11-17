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
from datetime import datetime
from collections import defaultdict
import json

# Torch imports
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# From own modules
from evaluators.segmentation_evaluator import SegmentationEvaluator
from plotting.training import TrainingVisualizer

# %%

class BaseTrainer(ABC):
    """
    Base trainer class handling common setup for semantic segmentation domain adaption experiments.    
    """
    def __init__(self, model, train_source_loader, train_target_loader, 
                 val_source_loader, val_target_loader, output_dir="outputs/experiment", 
                 device="cuda", num_classes=2, ignore_index=255,
                 monitor_metric="F1-Score"):
        self.model = model.to(device) # model
        
        # All Loaders used by trainer classes
        self.train_source_loader = train_source_loader
        self.train_target_loader = train_target_loader
        self.val_source_loader = val_source_loader
        self.val_target_loader = val_target_loader
        
        # General config
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Checkpoint tracking configuration
        self.monitor_metric = monitor_metric
        self.best_metric_value = -1.0
        
        # Config per run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # run timestamp
        self.run_dir = Path(f"{output_dir}_{timestamp}")
        self.log_dir = self.run_dir / "logs"
        self.save_dir = self.run_dir / "checkpoints"
        self.plot_dir = self.run_dir / "plots"
        
        for directory in (self.log_dir, self.save_dir, self.plot_dir): # Subdirectories creation
            directory.mkdir(parents=True, exist_ok=True)
        
        # Evaluator & Visualizer & TensorBoard Writer & Metrics dict & Initial config dict
        self.evaluator = SegmentationEvaluator(
            num_classes=num_classes, device=device, ignore_index=ignore_index
        )
        self.visualizer = TrainingVisualizer(plot_dir=self.plot_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.history = defaultdict(list)
        self.config = {
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "monitor_metric": monitor_metric
        }
        
    def log_scalar(self, name: str, value: float, epoch: int):
        """Logs a single scalar to both TensorBoard and in-memory history."""
        scalar_value = float(value) # Converts to float to enable saving in JSON
        self.writer.add_scalar(name, scalar_value, epoch) 
        self.history[name].append(scalar_value)

    def log_metrics(self, metrics: dict, epoch: int, prefix: str = "val"):
        """Logs a dictionary of metrics to both TensorBoard and in-memory history."""
        for key, value in metrics.items():
            self.log_scalar(f"{prefix}/{key}", value, epoch)

    def close(self):
        """Saves config, history, generates summary plots, and closes the TensorBoard writer."""
        self._save_config()
        self._save_history()
        self.visualizer.plot_training_summary(self.history, metric="mIoU", filename="summary_miou.png")
        self.visualizer.plot_training_summary(self.history, metric="F1-Score", filename="summary_f1.png")
        self.writer.close()
        
    def _evaluate_and_log(self, data_loader, prefix: str, epoch: int):
        """Universal method to evaluate any DataLoader, print results, and log them."""
        if data_loader is None:
            return None # returns None if the passed Data Loader is None
        
        metrics = self.evaluator.evaluate(self.model, data_loader)

        print(f"-> {prefix.capitalize()} mIoU: {metrics['mIoU']:.4f} | F1-Score: {metrics['F1-Score']:.4f}") # show in the Console
        
        self.log_metrics(metrics, epoch, prefix=prefix) # Log to TensorBoard an to history
        
        return metrics
        
    def _save_checkpoint(self, filename: str):
        """Helper method to save the model checkpoint."""
        filepath = self.save_dir / filename
        torch.save(self.model.state_dict(), filepath)
        print(f"[*] Checkpoint saved in: {filepath}")
        
    def _save_history(self, filename: str = "history.json"):
        """Saves the in-memory training history as a JSON file."""
        filepath = self.run_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dict(self.history), f, indent=4)
        print(f"[*] Training history saved in: {filepath}")
        
    def _save_config(self, filename: str = "config.json"):
        """Saves the completed configuration dictionary as a JSON file."""
        filepath = self.run_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
        print(f"[*] Experiment configuration saved in: {filepath}")
        
    def check_and_save_best_model(self, metrics: dict, epoch: int):
        """Centralized logic to check and save the best model checkpoint based on valid source metrics."""
        current_metric = metrics.get(self.monitor_metric, 0.0)
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            self._save_checkpoint("best_model.pth")
            print(f">>> [Checkpoint] New best model saved! ({self.monitor_metric}: {self.best_metric_value:.4f}) at epoch {epoch}")
        
    @abstractmethod
    def train(self, num_epochs: int):
        """
        Subclasses must implement the training loop with their domain adaptation strategy.
        """
        pass