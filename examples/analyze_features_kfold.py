"""
Example: Feature importance analysis across k-fold cross-validation.

This script analyzes feature importance across multiple folds to get more
robust estimates. It requires trained models from each fold.

If you don't have pretrained models from k-fold CV, you can:
1. First run train_kfold.py to train models and save them
2. Modify this script to train models on-the-fly (slower)
3. Or just use analyze_features.py for single-model analysis
"""

import sys
import os
import argparse

# Add parent directory to path to import rn_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from rn_analysis.config import DataConfig
from rn_analysis.dataloader import (
    list_csvs_by_class,
    get_file_labels,
    create_dataloaders,
    CycleDataset,
)
from rn_analysis.models import ImprovedCNN1D, ResNet1D, InceptionTime, RocketClassifier
from rn_analysis.train import train_model
from rn_analysis.utils import set_seed, get_device
from rn_analysis.feature_importance import (
    compute_permutation_importance,
    aggregate_importance_across_folds,
    plot_feature_importance,
    save_importance_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="K-fold feature importance analysis for time series classification"
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
        "--save-dir",
        type=str,
        default="feature_analysis_kfold",
        help="Directory to save results and models",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of permutation repeats per feature (per fold)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Do not train models if missing (error instead)",
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
        default=50,
        help="Number of epochs to train if models need to be trained",
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
    print("K-Fold Feature Importance Analysis")
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
    n_folds = args.n_folds
    n_repeats = args.n_repeats
    save_dir = args.save_dir
    train_if_missing = not args.no_train

    print("\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  K-folds: {n_folds}")
    print(f"  Data root: {data_config.root}")
    print(f"  Permutation repeats per fold: {n_repeats}")
    print(f"  Train if missing: {train_if_missing}")

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
    labels = get_file_labels(all_files, data_config.label_map)
    print(f"Total files: {len(all_files)}")

    # Get feature names and number of input channels
    df0 = pd.read_csv(all_files[0])
    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
    feature_cols = [c for c in df0.columns if c not in id_cols]
    input_channels = len(feature_cols)
    print(f"Input channels: {input_channels}")

    # -------------------------------------------------------------------------
    # Define model configuration
    # -------------------------------------------------------------------------
    model_configs = {
        "ImprovedCNN1D": {
            "class": ImprovedCNN1D,
            "kwargs": {"C": input_channels, "dropout": 0.3},
            "lr": 1e-3,
        },
        "ResNet1D": {
            "class": ResNet1D,
            "kwargs": {"input_channels": input_channels, "num_classes": 2, "dropout": 0.5},
            "lr": 1e-3,
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
            "lr": 1e-3,
        },
        "ROCKET": {
            "class": RocketClassifier,
            "kwargs": {
                "input_channels": input_channels,
                "num_classes": 2,
                "num_kernels": 5000,
                "dropout": 0.5,
            },
            "lr": 1e-3,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")

    config = model_configs[model_name]

    # -------------------------------------------------------------------------
    # K-Fold Feature Importance
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"# Computing Feature Importance Across {n_folds} Folds")
    print(f"{'#' * 80}\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=data_config.seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_files, labels)):
        print(f"\n{'-' * 80}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'-' * 80}")

        # Split files
        train_files = [all_files[i] for i in train_idx]
        test_files = [all_files[i] for i in test_idx]
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

        # Load or train model for this fold
        model_path = os.path.join(save_dir, f"{model_name}_fold{fold_idx}.pth")

        model = config["class"](**config["kwargs"])

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif train_if_missing:
            print(f"Training model for fold {fold_idx}...")
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=config["lr"],
                weight_decay=1e-2,
                device=str(device),
                early_stopping_patience=10,
            )
            # Save model
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path} and train_if_missing=False"
            )

        model = model.to(device)
        model.eval()

        # Compute permutation importance for this fold
        print(f"\nComputing permutation importance for fold {fold_idx}...")
        perm_results = compute_permutation_importance(
            model=model,
            dataloader=test_loader,
            feature_names=feature_cols,
            device=str(device),
            n_repeats=n_repeats,
            metric="accuracy",
        )

        fold_results.append(perm_results)

        # Print top features for this fold
        print(f"\nTop 5 features for fold {fold_idx}:")
        sorted_idx = perm_results["importances"].argsort()[::-1]
        for i, idx in enumerate(sorted_idx[:5]):
            print(
                f"  {i+1}. {feature_cols[idx]:<30s} "
                f"{perm_results['importances'][idx]:.4f}"
            )

    # -------------------------------------------------------------------------
    # Aggregate Results Across Folds
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print("# Aggregating Results Across Folds")
    print(f"{'#' * 80}\n")

    aggregated_results = aggregate_importance_across_folds(fold_results)

    print("Top 15 Most Important Features (Averaged Across Folds):")
    print("-" * 70)
    sorted_idx = aggregated_results["importances_mean"].argsort()[::-1]
    for i, idx in enumerate(sorted_idx[:15]):
        print(
            f"{i+1:2d}. {feature_cols[idx]:<30s} "
            f"{aggregated_results['importances_mean'][idx]:.4f} ± "
            f"{aggregated_results['importances_std'][idx]:.4f}"
        )

    # -------------------------------------------------------------------------
    # Visualize and Save Results
    # -------------------------------------------------------------------------
    print("\nSaving results...")

    # Plot aggregated importance
    plot_path = os.path.join(save_dir, f"{model_name}_kfold_importance.png")
    plot_feature_importance(
        aggregated_results,
        top_k=20,
        save_path=plot_path,
    )

    # Save to CSV
    csv_path = os.path.join(save_dir, f"{model_name}_kfold_importance.csv")
    save_importance_results(aggregated_results, csv_path)

    # Save per-fold results
    per_fold_df = pd.DataFrame({
        "feature": feature_cols,
    })
    for fold_idx, result in enumerate(fold_results):
        per_fold_df[f"fold_{fold_idx}_importance"] = result["importances"]

    per_fold_csv = os.path.join(save_dir, f"{model_name}_per_fold_importance.csv")
    per_fold_df.to_csv(per_fold_csv, index=False)
    print(f"Saved per-fold results to {per_fold_csv}")

    # Compute stability of feature importance across folds
    importance_matrix = np.array([r["importances"] for r in fold_results])
    stability_scores = 1 - (importance_matrix.std(axis=0) /
                           (importance_matrix.mean(axis=0) + 1e-10))

    stability_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_importance": aggregated_results["importances_mean"],
        "std_importance": aggregated_results["importances_std"],
        "stability_score": stability_scores,
    })
    stability_df = stability_df.sort_values("mean_importance", ascending=False)

    stability_csv = os.path.join(save_dir, f"{model_name}_stability.csv")
    stability_df.to_csv(stability_csv, index=False)
    print(f"Saved stability analysis to {stability_csv}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("K-FOLD ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {save_dir}/")
    print(f"  - Aggregated importance plot: {plot_path}")
    print(f"  - Aggregated importance CSV: {csv_path}")
    print(f"  - Per-fold results: {per_fold_csv}")
    print(f"  - Stability analysis: {stability_csv}")
    print("\nInterpretation:")
    print("  - High mean importance = feature is important for predictions")
    print("  - Low std importance = feature importance is consistent across folds")
    print("  - High stability score = feature is reliably important (recommended)")


if __name__ == "__main__":
    main()
