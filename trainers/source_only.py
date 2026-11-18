# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:40:40 2026

@author: Marcel
"""

# %% Imports

# From standard modules
import time

# From own modules
from .base_trainer import BaseTrainer

# From Pytorch
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
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
        
        self.config.update({
            "trainer": "SourceOnlyTrainer",
            "learning_rate": lr,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "scheduler": "PolynomialLR",
            "power": 0.9
        })
        
    def train(self, num_epochs: int):
        # PyTorch Polynomial LR Scheduler with standard power=0.9
        scheduler = PolynomialLR(self.optimizer, total_iters=num_epochs, power=0.9)
        
        print(f"[*] Starting training for {num_epochs} epochs. Monitoring metric: val_source/{self.monitor_metric}")
        
        # Start global training stopwatch
        self.start_timer()
        
        for epoch in range(num_epochs): # Train loop
            epoch_start_time = time.perf_counter()  # Start epoch stopwatch
        
            epoch_idx = epoch + 1
            self.model.train() # model in train mode
            total_loss = 0.0
            
            for batch in self.train_source_loader: # batch loop for the epoch
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                self.optimizer.zero_grad() # Zero gradients
                outputs = self.model(images) # Model predictions
                loss = self.criterion(outputs, masks) # Loss calculation
                loss.backward() # Backward propagation
                self.optimizer.step() # Optimizer step
                
                total_loss += loss.item() # Loss accumulation

            avg_loss = total_loss / len(self.train_source_loader) # Average the loss
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_duration = time.perf_counter() - epoch_start_time # Compute epoch duration
            
            print(f"[Source-Only] Epoch [{epoch_idx}/{num_epochs}] - Loss: {avg_loss:.4f} - LR: {current_lr:.6f} - Time: {epoch_duration:.2f}s")

            self.log_scalar("train/loss", avg_loss, epoch_idx) # Log training loss to TensorBoard
            self.log_scalar("train/lr", current_lr, epoch_idx) # Log current learning rate to TensorBoard
            self.log_scalar("train/epoch_time_sec", epoch_duration, epoch_idx)

            # Evaluation on validation source set
            source_metrics = self._evaluate_and_log(self.val_source_loader, prefix="val_source", epoch=epoch_idx)
            if source_metrics:
                self.check_and_save_best_model(source_metrics, epoch_idx) # Save checkpoint checking the monitor metric
            
            # Evaluation on validation target set (Monitoring only)
            self._evaluate_and_log(self.val_target_loader, prefix="val_target", epoch=epoch_idx)
            
            # Update learning rate scheduler at the end of epoch
            scheduler.step()

        # Save checkpoint at the end of training
        self._save_checkpoint("source_only_final.pth")
        print(f"[*] Training completed successfully! Best source {self.monitor_metric}: {self.best_metric_value:.4f}")
        
        self.close() # Close TensorBoard writer, save history and training plot



        