"""
Helper functions for visualization and utilities.

This module provides utility functions for data visualization, device
detection, and seed setting.
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device (auto-detect if not specified).

    Args:
        device: Device string ("cuda", "cpu", or None for auto-detect)

    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def visualize_run_cycle_csv(csv_path: str, figsize: tuple = (12, 6)) -> None:
    """Visualize sensor data from a single run/cycle CSV file.

    Creates multiple plots showing:
    1. All sensor channels vs time
    2. avg_* columns vs time (if available)
    3. Humidity and temperature vs time (if available)
    4. Section over time (if available)

    Args:
        csv_path: Path to CSV file
        figsize: Figure size for plots
    """
    g = pd.read_csv(csv_path).sort_values("relative_time_sec")
    time = g["relative_time_sec"].to_numpy()

    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "humidity", "temperature", "patient_id"}
    avg_cols = [c for c in g.columns if c.startswith("avg_")]
    sensor_cols = [c for c in g.columns if c not in id_cols and c not in avg_cols]

    # Extract metadata for title
    title_run = g["run_id"].iloc[0] if "run_id" in g.columns else "?"
    title_cycle = g["cycle"].iloc[0] if "cycle" in g.columns else "?"

    # 1) All sensor channels
    plt.figure(figsize=figsize)
    for c in sensor_cols:
        plt.plot(time, g[c].to_numpy(), alpha=0.25, linewidth=1)
    plt.title(f"Sensors vs time (run_id={title_run}, cycle={title_cycle})")
    plt.xlabel("relative_time_sec")
    plt.ylabel("sensor value")
    plt.tight_layout()
    plt.show()

    # 2) avg_* columns
    if avg_cols:
        plt.figure(figsize=figsize)
        for c in avg_cols:
            plt.plot(time, g[c].to_numpy(), linewidth=2, label=c)
        plt.title(f"avg_* vs time (run_id={title_run}, cycle={title_cycle})")
        plt.xlabel("relative_time_sec")
        plt.ylabel("avg value")
        plt.legend(ncol=5, fontsize=9)
        plt.tight_layout()
        plt.show()

    # 3) humidity + temperature
    if {"humidity", "temperature"}.issubset(g.columns):
        plt.figure(figsize=figsize)
        plt.plot(time, g["humidity"].to_numpy(), label="humidity")
        plt.plot(time, g["temperature"].to_numpy(), label="temperature")
        plt.title(f"humidity & temperature vs time (run_id={title_run}, cycle={title_cycle})")
        plt.xlabel("relative_time_sec")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 4) section (optional)
    if "section" in g.columns:
        codes, uniq = pd.factorize(g["section"])
        plt.figure(figsize=(figsize[0], figsize[1] * 0.5))
        plt.step(time, codes, where="post")
        plt.yticks(range(len(uniq)), uniq)
        plt.title("section over time")
        plt.xlabel("relative_time_sec")
        plt.tight_layout()
        plt.show()


def plot_training_history(history: dict, figsize: tuple = (15, 5)) -> None:
    """Plot training history (loss, accuracy, AUC).

    Args:
        history: Dictionary with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc'
        figsize: Figure size
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # AUC
    ax3.plot(epochs, history["val_auc"], label="Val AUC", linewidth=2, color="green")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AUC")
    ax3.set_title("Validation AUC")
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random')
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_kfold_results(results: dict, figsize: tuple = (15, 5)) -> None:
    """Plot k-fold cross-validation results.

    Args:
        results: Dictionary from stratified_kfold_cv() function
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    fold_numbers = [r["fold"] for r in results["fold_results"]]
    aucs = [r["best_val_auc"] for r in results["fold_results"]]
    accs = [r["best_val_acc"] for r in results["fold_results"]]

    # AUC per fold
    ax1.bar(fold_numbers, aucs, alpha=0.7, color="steelblue")
    ax1.axhline(y=results["mean_auc"], color='r', linestyle='--',
                label=f'Mean: {results["mean_auc"]:.4f}', linewidth=2)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("AUC")
    ax1.set_title("AUC per Fold")
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Accuracy per fold
    ax2.bar(fold_numbers, accs, alpha=0.7, color="coral")
    ax2.axhline(y=results["mean_acc"], color='r', linestyle='--',
                label=f'Mean: {results["mean_acc"]:.4f}', linewidth=2)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per Fold")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
