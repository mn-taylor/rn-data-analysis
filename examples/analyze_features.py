"""
Example: Feature importance analysis for trained models.

This script demonstrates how to analyze feature importance using:
1. Permutation importance (model-agnostic)
2. Integrated gradients (temporal feature importance)

You can run this on a single trained model or aggregate results across k-folds.
"""

import sys
import os
import argparse

# Add parent directory to path to import rn_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd

from rn_analysis.config import DataConfig
from rn_analysis.dataloader import (
    list_csvs_by_class,
    create_dataloaders,
    train_test_split_files,
    CycleDataset,
)
from rn_analysis.models import ImprovedCNN1D, ResNet1D, InceptionTime, RocketClassifier
from rn_analysis.train import train_model
from rn_analysis.utils import set_seed, get_device
from rn_analysis.feature_importance import (
    compute_permutation_importance,
    compute_temporal_importance,
    aggregate_importance_across_folds,
    plot_feature_importance,
    plot_temporal_importance,
    save_importance_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature importance analysis for time series classification models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ImprovedCNN1D",
        choices=["ImprovedCNN1D", "ResNet1D", "InceptionTime", "ROCKET"],
        help="Model architecture to analyze",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing POSITIVE/ and CONTROL/ subdirectories",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights (.pth file)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="feature_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of permutation repeats per feature",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Skip temporal importance analysis (faster)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train if no pretrained model provided",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Feature Importance Analysis")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    data_config = DataConfig(
        root=args.data_root,
        T=512,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Analysis configuration
    model_name = args.model
    use_pretrained = args.pretrained is not None
    pretrained_path = args.pretrained
    n_repeats = args.n_repeats
    compute_temporal = not args.no_temporal
    save_dir = args.save_dir

    print("\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Use pretrained: {use_pretrained}")
    print(f"  Data root: {data_config.root}")
    print(f"  Permutation repeats: {n_repeats}")
    print(f"  Compute temporal importance: {compute_temporal}")

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Set seed and device
    # -------------------------------------------------------------------------
    set_seed(data_config.seed)
    device = get_device()
    print(f"\nDevice: {device}")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    all_files = list_csvs_by_class(data_config.root)
    print(f"Total files: {len(all_files)}")

    # Get feature names and number of input channels
    df0 = pd.read_csv(all_files[0])
    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
    feature_cols = [c for c in df0.columns if c not in id_cols]
    input_channels = len(feature_cols)
    print(f"Input channels: {input_channels}")
    print(f"Feature names: {feature_cols[:5]}... (showing first 5)")

    # Split data
    train_files, test_files = train_test_split_files(
        all_files, train_split=0.8, seed=data_config.seed
    )
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_files,
        test_files,
        label_map=data_config.label_map,
        T=data_config.T,
        batch_size=data_config.batch_size,
        output_format="channels_first",
    )

    # -------------------------------------------------------------------------
    # Load or train model
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"# Setting up model: {model_name}")
    print(f"{'#' * 80}\n")

    # Define model
    model_configs = {
        "ImprovedCNN1D": {
            "class": ImprovedCNN1D,
            "kwargs": {"C": input_channels, "dropout": 0.3},
        },
        "ResNet1D": {
            "class": ResNet1D,
            "kwargs": {"input_channels": input_channels, "num_classes": 2, "dropout": 0.5},
        },
        "InceptionTime": {
            "class": InceptionTime,
            "kwargs": {
                "input_channels": input_channels,
                "num_classes": 2,
                "n_filters": 32,
                "depth": 6,
                "dropout": 0.5,
            },
        },
        "ROCKET": {
            "class": RocketClassifier,
            "kwargs": {
                "input_channels": input_channels,
                "num_classes": 2,
                "num_kernels": 5000,
                "dropout": 0.5,
            },
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    config = model_configs[model_name]
    model = config["class"](**config["kwargs"])

    if use_pretrained and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}...")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("Training model from scratch...")
        print("Note: For better results, train with k-fold CV first and load a pretrained model.")

        # Quick training for demo (adjust epochs as needed)
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=1e-3,
            weight_decay=1e-2,
            device=str(device),
            early_stopping_patience=5,
        )

        # Save trained model
        model_path = os.path.join(save_dir, f"{model_name}_trained.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved trained model to {model_path}")

    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 1. Permutation Feature Importance
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print("# Computing Permutation Feature Importance")
    print(f"{'#' * 80}\n")

    perm_results = compute_permutation_importance(
        model=model,
        dataloader=test_loader,
        feature_names=feature_cols,
        device=str(device),
        n_repeats=n_repeats,
        metric="accuracy",
    )

    print("\nTop 10 Most Important Features:")
    print("-" * 60)
    sorted_idx = perm_results["importances"].argsort()[::-1]
    for i, idx in enumerate(sorted_idx[:10]):
        print(
            f"{i+1:2d}. {feature_cols[idx]:<30s} "
            f"{perm_results['importances'][idx]:.4f} ± {perm_results['importances_std'][idx]:.4f}"
        )

    # Plot and save
    perm_plot_path = os.path.join(save_dir, f"{model_name}_permutation_importance.png")
    plot_feature_importance(
        perm_results,
        top_k=20,
        save_path=perm_plot_path,
    )

    perm_csv_path = os.path.join(save_dir, f"{model_name}_permutation_importance.csv")
    save_importance_results(perm_results, perm_csv_path)

    # -------------------------------------------------------------------------
    # 2. Temporal Feature Importance (Integrated Gradients)
    # -------------------------------------------------------------------------
    if compute_temporal:
        print(f"\n{'#' * 80}")
        print("# Computing Temporal Feature Importance")
        print(f"{'#' * 80}\n")

        temporal_results = compute_temporal_importance(
            model=model,
            dataloader=test_loader,
            feature_names=feature_cols,
            device=str(device),
            method="integrated_gradients",
            target_class=1,  # Positive class
        )

        # Plot and save
        temporal_plot_path = os.path.join(
            save_dir, f"{model_name}_temporal_importance.png"
        )
        plot_temporal_importance(
            temporal_results,
            top_k=15,
            save_path=temporal_plot_path,
        )

        # Save temporal importance as numpy array
        import numpy as np
        temporal_npy_path = os.path.join(
            save_dir, f"{model_name}_temporal_importance.npy"
        )
        np.save(temporal_npy_path, temporal_results["temporal_importance"])
        print(f"Saved temporal importance array to {temporal_npy_path}")

        # Also save summary statistics
        temporal_importance = temporal_results["temporal_importance"]
        temporal_summary = pd.DataFrame({
            "feature": feature_cols,
            "mean_temporal_importance": temporal_importance.mean(axis=1),
            "max_temporal_importance": temporal_importance.max(axis=1),
            "std_temporal_importance": temporal_importance.std(axis=1),
        })
        temporal_summary = temporal_summary.sort_values(
            "mean_temporal_importance", ascending=False
        )
        temporal_csv_path = os.path.join(
            save_dir, f"{model_name}_temporal_importance_summary.csv"
        )
        temporal_summary.to_csv(temporal_csv_path, index=False)
        print(f"Saved temporal importance summary to {temporal_csv_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {save_dir}/")
    print(f"  - Permutation importance: {perm_plot_path}")
    print(f"  - Permutation CSV: {perm_csv_path}")
    if compute_temporal:
        print(f"  - Temporal importance: {temporal_plot_path}")
        print(f"  - Temporal CSV: {temporal_csv_path}")
    print("\nYou can now:")
    print("  1. Review the plots to identify key features")
    print("  2. Use the CSV files for further analysis")
    print("  3. Run this script on different models to compare feature importance")
    print("  4. Modify the script to analyze k-fold CV results")


if __name__ == "__main__":
    main()
