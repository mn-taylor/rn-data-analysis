"""
Configuration dataclasses for experiments.

This module provides type-safe configuration classes for all aspects of
the training pipeline, including data loading, model architecture, and
training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Args:
        root: Root directory containing class subdirectories
        T: Fixed time series length (resampled)
        feature_cols: List of feature column names (None = auto-detect)
        label_map: Mapping from folder names to class indices
        train_split: Fraction of data for training (rest is test)
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
    """
    root: str = "data"
    T: int = 512
    feature_cols: Optional[List[str]] = None
    label_map: Dict[str, int] = field(default_factory=lambda: {"CONTROL": 0, "POSITIVE": 1})
    train_split: float = 0.8
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for training process.

    Args:
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization weight
        grad_clip: Gradient clipping value (None = no clipping)
        early_stopping_patience: Epochs to wait before early stopping (None = no early stopping)
        device: Device to train on ("cuda", "cpu", or None for auto-detect)
        checkpoint_path: Path to save best model checkpoint
        print_every: Print metrics every N epochs (0 = only at end)
    """
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: Optional[float] = 1.0
    early_stopping_patience: Optional[int] = 10
    device: Optional[str] = None  # None = auto-detect
    checkpoint_path: str = "best_model.pt"
    print_every: int = 1


@dataclass
class ModelConfig:
    """Base configuration for all models.

    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    num_classes: int = 2
    dropout: float = 0.5


@dataclass
class CNNConfig(ModelConfig):
    """Configuration for ImprovedCNN1D model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout probability for classification head
    """
    dropout: float = 0.3


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for TimeSeriesTransformer model.

    Args:
        num_classes: Number of output classes
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_ff: Feedforward dimension
        dropout: Dropout probability
    """
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 256
    dropout: float = 0.1


@dataclass
class ResNetConfig(ModelConfig):
    """Configuration for ResNet1D model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout probability for classification head
    """
    dropout: float = 0.5


@dataclass
class InceptionConfig(ModelConfig):
    """Configuration for InceptionTime model.

    Args:
        num_classes: Number of output classes
        n_filters: Number of filters per inception branch
        depth: Number of inception modules
        kernel_sizes: Kernel sizes for parallel convolutions
        bottleneck_channels: Bottleneck channels (0 = no bottleneck)
        dropout: Dropout probability for classification head
    """
    n_filters: int = 32
    depth: int = 6
    kernel_sizes: List[int] = field(default_factory=lambda: [9, 19, 39])
    bottleneck_channels: int = 32
    dropout: float = 0.5


@dataclass
class RocketConfig(ModelConfig):
    """Configuration for ROCKET classifier.

    Args:
        num_classes: Number of output classes
        num_kernels: Number of random convolutional kernels
        dropout: Dropout probability for classification head
    """
    num_kernels: int = 5000
    dropout: float = 0.5


@dataclass
class KFoldConfig:
    """Configuration for K-fold cross-validation.

    Args:
        n_folds: Number of folds
        shuffle: Whether to shuffle before splitting
        stratified: Whether to use stratified splitting
    """
    n_folds: int = 5
    shuffle: bool = True
    stratified: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Combines data, training, and model configurations for a complete
    experiment specification.

    Args:
        name: Experiment name (for logging)
        data: Data configuration
        training: Training configuration
        model: Model configuration
        kfold: K-fold configuration (None = single train/test split)
    """
    name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=CNNConfig)
    kfold: Optional[KFoldConfig] = None

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = [f"Experiment: {self.name}"]
        lines.append("\nData Config:")
        for key, val in self.data.__dict__.items():
            lines.append(f"  {key}: {val}")
        lines.append("\nTraining Config:")
        for key, val in self.training.__dict__.items():
            lines.append(f"  {key}: {val}")
        lines.append("\nModel Config:")
        for key, val in self.model.__dict__.items():
            lines.append(f"  {key}: {val}")
        if self.kfold:
            lines.append("\nK-Fold Config:")
            for key, val in self.kfold.__dict__.items():
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)
