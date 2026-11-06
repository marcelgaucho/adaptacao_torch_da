# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:40:16 2026

@author: Marcel
"""

# %% Imports

# From Pytorch
import torch
import torch.nn as nn
from torchmetrics.classification import (MulticlassJaccardIndex, 
                                         MulticlassF1Score, 
                                         MulticlassPrecision,
                                         MulticlassRecall)


# %%

class SegmentationEvaluator:
    """
    Evaluator class for semantic segmentation tasks.
    Computes standard metrics including mIoU, F1-Score, Precision, and Recall.
    """
    def __init__(self, num_classes: int, device: str):
        self.num_classes = num_classes
        self.device = device
        
        # Mean Metrics (IoU, F1, Precision and Recall)
        self.miou = MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
        
    def evaluate(self, model: nn.Module, val_loader) -> dict:
        """
        Evaluates the model on a given validation or test loader.
        """
        model.eval()
        
        # Reset all metric states before starting evaluation
        self.miou.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass and prediction extraction
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Update metric states with current batch predictions and targets
                self.miou.update(preds, masks)
                self.f1.update(preds, masks)
                self.precision.update(preds, masks)
                self.recall.update(preds, masks)
                
        # Compute and return final scalar metrics
        return {
            "mIoU": self.miou.compute().item(),
            "F1-Score": self.f1.compute().item(),
            "Precision": self.precision.compute().item(),
            "Recall": self.recall.compute().item()
        }
    

