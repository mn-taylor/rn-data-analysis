"""
RN Data Analysis - Modular Time Series Classification

A modular framework for time series classification on medical sensor data.

Modules:
    config: Configuration dataclasses for experiments
    dataloader: Dataset classes and data loading utilities
    models: Neural network model architectures
    train: Training, validation, and evaluation functions
    utils: Visualization and helper utilities
"""

__version__ = "0.1.0"

from . import config
from . import dataloader
from . import models
from . import train
from . import utils

__all__ = [
    "config",
    "dataloader",
    "models",
    "train",
    "utils",
]
