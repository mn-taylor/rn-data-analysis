"""
Example: K-fold cross-validation with multiple models.

This script demonstrates using stratified k-fold cross-validation to
evaluate and compare different model architectures.
"""

import sys
import os

# Add parent directory to path to import rn_analysis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from rn_analysis.config import DataConfig
from rn_analysis.dataloader import list_csvs_by_class
from rn_analysis.models import ImprovedCNN1D, ResNet1D, InceptionTime, RocketClassifier
from rn_analysis.train import stratified_kfold_cv
from rn_analysis.utils import set_seed, get_device, plot_kfold_results


def main():
    print("=" * 80)
    print("K-Fold Cross-Validation: Model Comparison")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    data_config = DataConfig(
        root="data",
        T=512,
        batch_size=32,
        seed=42,
    )

    # K-fold parameters
    n_folds = 5
    epochs = 50
    early_stopping_patience = 10

    print("\nData Config:")
    print(f"  Root: {data_config.root}")
    print(f"  T: {data_config.T}")
    print(f"  Batch size: {data_config.batch_size}")
    print(f"  K-folds: {n_folds}")
    print(f"  Max epochs per fold: {epochs}")

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

    # Get number of input channels from first file
    import pandas as pd
    df0 = pd.read_csv(all_files[0])
    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
    feature_cols = [c for c in df0.columns if c not in id_cols]
    input_channels = len(feature_cols)
    print(f"Input channels: {input_channels}")

    # -------------------------------------------------------------------------
    # Define models to evaluate
    # -------------------------------------------------------------------------
    model_configs = {
        "ImprovedCNN1D": {
            "class": ImprovedCNN1D,
            "kwargs": {"C": input_channels, "dropout": 0.3},
            "lr": 1e-3,
            "batch_size": 32,
        },
        "ResNet1D": {
            "class": ResNet1D,
            "kwargs": {"input_channels": input_channels, "num_classes": 2, "dropout": 0.5},
            "lr": 1e-3,
            "batch_size": 32,
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
            "batch_size": 32,
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
            "batch_size": 32,
        },
    }

    print("\nModels to evaluate:")
    for name in model_configs.keys():
        print(f"  - {name}")

    # -------------------------------------------------------------------------
    # Run K-Fold CV for each model
    # -------------------------------------------------------------------------
    all_results = {}

    for model_name, config in model_configs.items():
        print(f"\n{'#' * 80}")
        print(f"# Evaluating: {model_name}")
        print(f"{'#' * 80}\n")

        results = stratified_kfold_cv(
            model_class=config["class"],
            model_kwargs=config["kwargs"],
            files=all_files,
            label_map=data_config.label_map,
            n_folds=n_folds,
            T=data_config.T,
            batch_size=config["batch_size"],
            epochs=epochs,
            lr=config["lr"],
            weight_decay=1e-2,
            device=str(device),
            seed=data_config.seed,
            early_stopping_patience=early_stopping_patience,
            output_format="channels_first",
        )

        all_results[model_name] = results

        # Plot results for this model
        try:
            print(f"\nPlotting results for {model_name}...")
            plot_kfold_results(results)
        except Exception as e:
            print(f"Could not plot results: {e}")

    # -------------------------------------------------------------------------
    # Compare all models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - ALL MODELS")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean AUC':<15} {'Std AUC':<15} {'Mean Acc':<15} {'Std Acc':<15}")
    print("-" * 80)

    for model_name, results in all_results.items():
        print(
            f"{model_name:<20} "
            f"{results['mean_auc']:.4f} ± {results['std_auc']:.4f}   "
            f"{results['mean_acc']:.4f} ± {results['std_acc']:.4f}"
        )

    # Find best model by AUC
    best_model = max(all_results.items(), key=lambda x: x[1]["mean_auc"])
    print("=" * 80)
    print(f"BEST MODEL: {best_model[0]} (Mean AUC: {best_model[1]['mean_auc']:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
