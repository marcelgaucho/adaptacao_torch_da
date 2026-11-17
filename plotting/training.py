# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:56:43 2026

@author: Marcel
"""

# %% Imports

# Standard Imports
from pathlib import Path

# Other imports
import matplotlib.pyplot as plt

# %%

class TrainingVisualizer:
    """Handles generation and saving of training summary plots."""
    
    def __init__(self, plot_dir: Path):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_summary(self, history: dict, metric: str = "mIoU",
                              filename: str = "training_summary.png"):
        """Generates and saves a 2-panel summary plot (Losses & Domain Adaptation Gap)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        self._plot_losses(ax1, history) # Plot loss in first subplot
        self._plot_domain_gap(ax2, history, metric=metric) # Plot domain gap in second subplot
        
        plt.tight_layout()
        plot_path = self.plot_dir / filename # complete figure path
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"[*] Training summary plot saved in: {plot_path}")
        
    def _plot_losses(self, ax, history: dict):
        has_curves = False
        for key, values in history.items():
            if "loss" in key:
                ax.plot(range(1, len(values) + 1), values, label=key, linewidth=2)
                has_curves = True
                
        ax.set_title("Training Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        if has_curves:
            ax.legend()
            
    def _plot_domain_gap(self, ax, history: dict, metric: str):
        source_key = f"val_source/{metric}"
        target_key = f"val_target/{metric}"
        has_curves = False
        
        if source_key in history:
            epochs = range(1, len(history[source_key]) + 1)
            ax.plot(epochs, history[source_key], label=f"Source Val {metric}", linestyle="--", linewidth=2)
            has_curves = True
        if target_key in history:
            epochs = range(1, len(history[target_key]) + 1)
            ax.plot(epochs, history[target_key], label=f"Target Val {metric}", linewidth=2.5)
            has_curves = True
            
        ax.set_title(f"Domain Adaptation Gap ({metric})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        if has_curves:
            ax.legend()



