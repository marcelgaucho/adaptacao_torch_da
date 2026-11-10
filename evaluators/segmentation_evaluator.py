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
    def __init__(self, num_classes: int, device: str, ignore_index: int = 255):
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = ignore_index
        self.is_binary = (num_classes == 2)
        
        avg_mode = "none" if self.is_binary else "macro" # Average metrics for all classes if there is more than 2 classes
        
        # Metrics (IoU, F1, Precision and Recall)
        self.miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=avg_mode).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average=avg_mode).to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, ignore_index=ignore_index, average=avg_mode).to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, ignore_index=ignore_index, average=avg_mode).to(device)
        
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
                
        # Compute metrics
        miou_val = self.miou.compute()
        f1_val = self.f1.compute()
        precision_val = self.precision.compute()
        recall_val = self.recall.compute()
                
        # Only the positive class is used when num_classes = 2
        if self.is_binary:
            return {
                "mIoU": miou_val[1].item(),
                "F1-Score": f1_val[1].item(),
                "Precision": precision_val[1].item(),
                "Recall": recall_val[1].item()
            }
        else:
            return {
                "mIoU": miou_val.item(),
                "F1-Score": f1_val.item(),
                "Precision": precision_val.item(),
                "Recall": recall_val.item()
            }


