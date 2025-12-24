"""
Example: Train ImprovedCNN1D with 80/20 train/test split.

This script demonstrates basic usage of the rn_analysis package for
training a CNN model on time series data.
"""

import sys
import os

# Add parent directory to path to import rn_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from rn_analysis.config import DataConfig, TrainingConfig, CNNConfig
from rn_analysis.dataloader import list_csvs_by_class, train_test_split_files, create_dataloaders
from rn_analysis.models import ImprovedCNN1D
from rn_analysis.train import train_model
from rn_analysis.utils import set_seed, get_device, plot_training_history, count_parameters


def main():
    print("=" * 80)
    print("Training ImprovedCNN1D on Time Series Data")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    data_config = DataConfig(
        root="data",
        T=512,
        train_split=0.8,
        batch_size=32,
        seed=42,
    )

    training_config = TrainingConfig(
        epochs=50,
        lr=1e-3,
        weight_decay=1e-2,
        grad_clip=1.0,
        early_stopping_patience=10,
        checkpoint_path="models/cnn_best.pt",
        print_every=1,
    )

    model_config = CNNConfig(
        num_classes=2,
        dropout=0.3,
    )

    print("\nData Config:")
    print(f"  Root: {data_config.root}")
    print(f"  T: {data_config.T}")
    print(f"  Batch size: {data_config.batch_size}")
    print(f"  Train split: {data_config.train_split}")

    print("\nTraining Config:")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Learning rate: {training_config.lr}")
    print(f"  Weight decay: {training_config.weight_decay}")
    print(f"  Early stopping patience: {training_config.early_stopping_patience}")

    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    # -------------------------------------------------------------------------
    set_seed(data_config.seed)
    device = get_device(training_config.device)
    print(f"\nDevice: {device}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    all_files = list_csvs_by_class(data_config.root)
    print(f"Total files: {len(all_files)}")

    # Split into train/test
    train_files, test_files = train_test_split_files(
        all_files, train_split=data_config.train_split, seed=data_config.seed
    )
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_files,
        test_files,
        label_map=data_config.label_map,
        T=data_config.T,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        output_format="channels_first",  # CNN expects (B, C, T)
    )

    # Get number of input channels from first batch
    x_sample, _ = next(iter(train_loader))
    input_channels = x_sample.shape[1]
    print(f"Input channels: {input_channels}")

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    print("\nCreating model...")
    model = ImprovedCNN1D(C=input_channels, dropout=model_config.dropout).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    print("\nStarting training...")
    print("=" * 80)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=training_config.epochs,
        grad_clip=training_config.grad_clip,
        early_stopping_patience=training_config.early_stopping_patience,
        checkpoint_path=training_config.checkpoint_path,
        print_every=training_config.print_every,
        use_tqdm=True,
    )

    # -------------------------------------------------------------------------
    # Visualize results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    try:
        plot_training_history(history)
    except Exception as e:
        print(f"Could not plot history: {e}")


if __name__ == "__main__":
    main()
