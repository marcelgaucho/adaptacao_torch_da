# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:40:40 2026

@author: Marcel
"""

# %% Imports

# From standard modules
...

# From own modules
from .base_trainer import BaseTrainer

# From Pytorch
import torch.optim as optim
import torch.nn as nn

# %%

class SourceOnlyTrainer(BaseTrainer):
    """
    Baseline method: Trains the segmentation model exclusively on the source domain.
    """
    def __init__(self, model, train_source_loader, train_target_loader, 
                val_source_loader, val_target_loader, lr=1e-3, **kwargs):
        super().__init__(model, train_source_loader, train_target_loader, 
                        val_source_loader, val_target_loader, **kwargs)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)       
        
    def train(self, num_epochs: int):
        # Loop through the epochs
        for epoch in range(num_epochs):
            self.model.train() # model in train mode
            total_loss = 0.0
            
            for images, masks in self.train_source_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad() # Zero gradients
                outputs = self.model(images) # Model predictions
                loss = self.criterion(outputs, masks) # Loss calculation
                loss.backward() # Backward propagation
                self.optimizer.step() # Optimizer step
                
                total_loss += loss.item() # Loss accumulation

            avg_loss = total_loss / len(self.train_source_loader) # Average the loss
            print(f"[Source-Only] Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

            self.writer.add_scalar("train/loss", avg_loss, epoch + 1) # Log training loss to TensorBoard

            # Evaluation on validation source set
            if self.val_source_loader is not None:
                source_metrics = self.evaluator.evaluate(self.model, self.val_source_loader)
                print(f"-> Source Validation mIoU: {source_metrics['mIoU']:.4f}")
                print(f"-> Source Validation F1-Score: {source_metrics['F1-Score']:.4f}")

                # Log tracked metrics to TensorBoard
                self.log_metrics(source_metrics, epoch + 1, prefix="val_source")

            # Evaluation on validation target set (Monitoring only)
            if self.val_target_loader is not None:
                target_metrics = self.evaluator.evaluate(self.model, self.val_target_loader)
                print(f"-> Target Validation mIoU: {target_metrics['mIoU']:.4f}")
                print(f"-> Target Validation F1-Score: {target_metrics['F1-Score']:.4f}")

                # Log tracked metrics to TensorBoard
                self.log_metrics(target_metrics, epoch + 1, prefix="val_target")

        # Save checkpoint in the end of training
        self._save_checkpoint("source_only_final.pth")
        
        self.close() # Close TensorBoard writer