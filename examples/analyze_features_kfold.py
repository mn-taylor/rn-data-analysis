"""
K-fold feature importance analysis — config-driven entry point.

Usage
-----
    python examples/analyze_features_kfold.py \\
        --config examples/configs/analyze_kfold.yaml

All behaviour is controlled by the YAML config file.  See
examples/configs/analyze_kfold.yaml for a fully annotated template.

Outputs (written to output.save_dir)
-------------------------------------
  {model}_fold{k}.pth                  — per-fold model checkpoints
  {model}_kfold_importance.png/csv     — aggregated permutation importance
  {model}_per_fold_importance.csv      — raw per-fold permutation scores
  {model}_stability.csv                — CV stability scores
  {model}_all_methods.png              — side-by-side multi-method plot
  {model}_all_methods.csv             — per-feature scores for all methods
  {model}_method_correlations.csv     — Spearman ρ matrix
  {model}_method_pvalues.csv          — p-value matrix
  {model}_method_corr_heatmap.png     — Spearman ρ heatmap
  {model}_ablation.png                — channel group ablation bar chart
  {model}_ablation_results.csv        — ablation scores per condition
  {model}_channel_groups.csv          — channel-to-group membership
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from rn_analysis.config import DataConfig
from rn_analysis.dataloader import (
    list_csvs_by_class,
    get_file_labels,
    create_dataloaders,
)
from rn_analysis.models import (
    ImprovedCNN1D,
    ResNet1D,
    InceptionTime,
    RocketClassifier,
)
from rn_analysis.train import train_model
from rn_analysis.utils import set_seed, get_device
from rn_analysis.feature_importance import (
    compute_permutation_importance,
    compute_occlusion_importance,
    compute_ig_channel_importance,
    compute_shap_importance,
    aggregate_importance_across_folds,
    compare_importance_methods,
    plot_feature_importance,
    plot_multi_method_importance,
    plot_method_comparison,
    save_importance_results,
    save_method_comparison,
)
from rn_analysis.channel_ablation import (
    identify_channel_groups,
    run_channel_ablation,
    aggregate_ablation_across_folds,
    plot_ablation_results,
    save_ablation_results,
)


# =============================================================================
# Config helpers
# =============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Model factory
# =============================================================================

def build_model(model_name: str, input_channels: int, model_cfg: dict):
    """Instantiate the model specified in the config."""
    dropout = model_cfg.get("dropout", 0.3)

    if model_name == "ImprovedCNN1D":
        return ImprovedCNN1D(C=input_channels, dropout=dropout)

    elif model_name == "ResNet1D":
        return ResNet1D(
            input_channels=input_channels,
            num_classes=2,
            dropout=dropout,
        )

    elif model_name == "InceptionTime":
        return InceptionTime(
            input_channels=input_channels,
            num_classes=2,
            n_filters=model_cfg.get("n_filters", 32),
            depth=model_cfg.get("depth", 6),
            dropout=dropout,
        )

    elif model_name == "ROCKET":
        return RocketClassifier(
            input_channels=input_channels,
            num_classes=2,
            num_kernels=model_cfg.get("num_kernels", 5000),
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")


# =============================================================================
# Model loading / training
# =============================================================================

def get_or_train_model(
    model,
    model_path: str,
    train_loader,
    val_loader,
    training_cfg: dict,
    device,
):
    """Load model from checkpoint, or train it if the checkpoint is missing."""
    if os.path.exists(model_path):
        print(f"  Loading checkpoint: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    if not training_cfg.get("train_if_missing", True):
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path!r} "
            "and train_if_missing is False."
        )

    print(f"  Checkpoint not found — training model...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("lr", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-2),
    )
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
        epochs=training_cfg.get("epochs", 50),
        early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
        checkpoint_path=model_path,
    )
    print(f"  Saved checkpoint: {model_path}")
    return model


# =============================================================================
# Importance methods dispatcher
# =============================================================================

def _supports_gradients(model_name: str) -> bool:
    """ROCKET uses non-differentiable PPV — skip gradient-based methods."""
    return model_name != "ROCKET"


def run_importance_methods(
    model,
    test_loader,
    feature_names,
    analysis_cfg: dict,
    device,
    model_name: str,
) -> dict:
    """Run all enabled importance methods and return a results dict."""
    methods  = analysis_cfg.get("methods", ["permutation"])
    metric   = analysis_cfg.get("metric", "accuracy")
    results  = {}

    if "permutation" in methods:
        print("\n  [Permutation Importance]")
        results["permutation"] = compute_permutation_importance(
            model, test_loader, feature_names, device,
            n_repeats=analysis_cfg.get("n_repeats", 5),
            metric=metric,
        )

    if "occlusion" in methods:
        print("\n  [Occlusion Importance]")
        results["occlusion"] = compute_occlusion_importance(
            model, test_loader, feature_names, device,
            metric=metric,
        )

    if "integrated_gradients" in methods:
        if _supports_gradients(model_name):
            print("\n  [Integrated Gradients]")
            try:
                results["integrated_gradients"] = compute_ig_channel_importance(
                    model, test_loader, feature_names, device,
                    n_steps=analysis_cfg.get("ig_n_steps", 50),
                    target_class=analysis_cfg.get("ig_target_class", 1),
                    n_samples=analysis_cfg.get("ig_n_samples", None),
                )
            except Exception as exc:
                print(f"  [Integrated Gradients] Failed: {exc}")
        else:
            print(f"\n  [Integrated Gradients] Skipped — not supported for {model_name}")

    if "shap" in methods:
        if _supports_gradients(model_name):
            print("\n  [SHAP]")
            try:
                results["shap"] = compute_shap_importance(
                    model, test_loader, feature_names, device,
                    n_background=analysis_cfg.get("shap_n_background", 50),
                    n_samples=analysis_cfg.get("shap_n_samples", 100),
                    target_class=analysis_cfg.get("shap_target_class", 1),
                )
            except Exception as exc:
                print(f"  [SHAP] Failed: {exc}")
        else:
            print(f"\n  [SHAP] Skipped — not supported for {model_name}")

    return results


# =============================================================================
# Aggregation helpers
# =============================================================================

def aggregate_methods_across_folds(
    fold_method_results: list,
) -> dict:
    """For each method, aggregate importances across folds.

    Returns dict: method_name -> aggregated importance dict
    (with importances_mean / importances_std / feature_names).
    """
    # Collect method names from all folds (some may be missing if a fold failed)
    all_methods = set()
    for fold_res in fold_method_results:
        all_methods.update(fold_res.keys())

    aggregated = {}
    for method in all_methods:
        fold_dicts = [
            fold_res[method]
            for fold_res in fold_method_results
            if method in fold_res
        ]
        if fold_dicts:
            aggregated[method] = aggregate_importance_across_folds(fold_dicts)

    return aggregated


# =============================================================================
# Saving helpers
# =============================================================================

def save_permutation_per_fold(
    fold_method_results: list,
    feature_cols: list,
    save_dir: str,
    model_name: str,
):
    """Write per-fold permutation importance to CSV (for stability analysis)."""
    per_fold_df = pd.DataFrame({"feature": feature_cols})
    for fold_idx, fold_res in enumerate(fold_method_results):
        if "permutation" in fold_res:
            per_fold_df[f"fold_{fold_idx}_importance"] = (
                fold_res["permutation"]["importances"]
            )
    path = os.path.join(save_dir, f"{model_name}_per_fold_importance.csv")
    per_fold_df.to_csv(path, index=False)
    print(f"Saved per-fold permutation results to {path}")


def save_stability(
    fold_method_results: list,
    feature_cols: list,
    aggregated_methods: dict,
    save_dir: str,
    model_name: str,
):
    """Write CV stability analysis (coefficient of variation) to CSV."""
    if "permutation" not in aggregated_methods:
        return

    importance_matrix = np.array(
        [
            fold_res["permutation"]["importances"]
            for fold_res in fold_method_results
            if "permutation" in fold_res
        ]
    )
    stability = 1 - (
        importance_matrix.std(axis=0)
        / (importance_matrix.mean(axis=0) + 1e-10)
    )

    agg = aggregated_methods["permutation"]
    stability_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_importance": agg["importances_mean"],
            "std_importance": agg["importances_std"],
            "stability_score": stability,
        }
    ).sort_values("mean_importance", ascending=False)

    path = os.path.join(save_dir, f"{model_name}_stability.csv")
    stability_df.to_csv(path, index=False)
    print(f"Saved stability analysis to {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="K-fold feature importance analysis (config-driven)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/analyze_kfold.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    kfold_cfg    = cfg["kfold"]
    analysis_cfg = cfg["analysis"]
    groups_cfg   = cfg.get("channel_groups", {})
    output_cfg   = cfg["output"]

    model_name = model_cfg["name"]
    n_folds    = kfold_cfg["n_folds"]
    save_dir   = output_cfg["save_dir"]
    top_k      = output_cfg.get("top_k", 20)
    metric     = analysis_cfg.get("metric", "accuracy")

    print("=" * 80)
    print("K-Fold Feature Importance Analysis")
    print("=" * 80)
    print(f"\n  Model      : {model_name}")
    print(f"  K-folds    : {n_folds}")
    print(f"  Data root  : {data_cfg['root']}")
    print(f"  Methods    : {analysis_cfg.get('methods', [])}")
    print(f"  Save dir   : {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    set_seed(data_cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))
    print(f"\n  Device: {device}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    data_config = DataConfig(
        root=data_cfg["root"],
        T=data_cfg.get("T", 512),
        batch_size=data_cfg.get("batch_size", 32),
        seed=data_cfg.get("seed", 42),
    )

    all_files = list_csvs_by_class(data_config.root)
    labels    = get_file_labels(all_files, data_config.label_map)
    print(f"  Total files: {len(all_files)}")

    # Detect feature columns from first file
    df0 = pd.read_csv(all_files[0])
    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
    feature_cols   = [c for c in df0.columns if c not in id_cols]
    input_channels = len(feature_cols)
    print(f"  Input channels: {input_channels}")

    # Identify channel groups for ablation
    print("\nChannel groups:")
    group_indices = identify_channel_groups(feature_cols, groups_cfg)

    # -------------------------------------------------------------------------
    # K-Fold loop
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"# K-Fold Analysis ({n_folds} folds)")
    print(f"{'#' * 80}")

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=data_config.seed,
    )

    fold_method_results: list = []   # list of method->result dicts
    fold_ablation_results: list = [] # list of (results, display_names) tuples

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(all_files, labels)
    ):
        print(f"\n{'-' * 80}")
        print(f"FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'-' * 80}")

        train_files = [all_files[i] for i in train_idx]
        test_files  = [all_files[i] for i in test_idx]
        print(f"  Train: {len(train_files)}  Test: {len(test_files)}")

        train_loader, test_loader = create_dataloaders(
            train_files,
            test_files,
            label_map=data_config.label_map,
            T=data_config.T,
            batch_size=data_config.batch_size,
            output_format="channels_first",
        )

        # Load or train model
        model_path = os.path.join(
            save_dir, f"{model_name}_fold{fold_idx}.pth"
        )
        model = build_model(model_name, input_channels, model_cfg)
        model = get_or_train_model(
            model, model_path, train_loader, test_loader,
            training_cfg, device,
        )
        model = model.to(device)
        model.eval()

        # -- Importance methods --
        print(f"\n  Computing importance methods for fold {fold_idx + 1}...")
        fold_res = run_importance_methods(
            model, test_loader, feature_cols, analysis_cfg, device, model_name
        )
        fold_method_results.append(fold_res)

        # Print top-5 from permutation (quick sanity check)
        if "permutation" in fold_res:
            top5 = fold_res["permutation"]["importances"].argsort()[::-1][:5]
            print(f"\n  Top-5 (permutation):")
            for rank, idx in enumerate(top5):
                print(
                    f"    {rank+1}. {feature_cols[idx]:<30s}"
                    f" {fold_res['permutation']['importances'][idx]:.4f}"
                )

        # -- Channel ablation --
        if groups_cfg:
            print(f"\n  Running channel ablation for fold {fold_idx + 1}...")
            abl_res, abl_names = run_channel_ablation(
                model, test_loader, feature_cols,
                group_indices, groups_cfg, device, metric=metric,
            )
            fold_ablation_results.append((abl_res, abl_names))

    # -------------------------------------------------------------------------
    # Aggregate across folds
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print("# Aggregating Results")
    print(f"{'#' * 80}\n")

    aggregated_methods = aggregate_methods_across_folds(fold_method_results)

    # -------------------------------------------------------------------------
    # Print leaderboard (permutation)
    # -------------------------------------------------------------------------
    if "permutation" in aggregated_methods:
        agg_perm = aggregated_methods["permutation"]
        print("Top 15 features (permutation, mean across folds):")
        print("-" * 70)
        top15 = agg_perm["importances_mean"].argsort()[::-1][:15]
        for rank, idx in enumerate(top15):
            print(
                f"  {rank+1:2d}. {feature_cols[idx]:<30s}"
                f" {agg_perm['importances_mean'][idx]:.4f}"
                f" ± {agg_perm['importances_std'][idx]:.4f}"
            )

    # -------------------------------------------------------------------------
    # Spearman comparison between methods
    # -------------------------------------------------------------------------
    if len(aggregated_methods) > 1:
        print("\nSpearman ρ between importance methods:")
        corr_matrix, pval_matrix, method_names = compare_importance_methods(
            aggregated_methods
        )
        for i, ni in enumerate(method_names):
            for j, nj in enumerate(method_names):
                if i < j:
                    print(
                        f"  {ni} vs {nj}: ρ = {corr_matrix[i,j]:.3f}"
                        f"  (p = {pval_matrix[i,j]:.3f})"
                    )
    else:
        corr_matrix = pval_matrix = method_names = None

    # -------------------------------------------------------------------------
    # Aggregate ablation
    # -------------------------------------------------------------------------
    if fold_ablation_results:
        agg_ablation, abl_display_names = aggregate_ablation_across_folds(
            fold_ablation_results
        )
        print("\nAblation summary (mean ± std across folds):")
        for cond, (mean, std) in agg_ablation.items():
            print(
                f"  {abl_display_names.get(cond, cond):<35s}"
                f"  {mean:.4f} ± {std:.4f}"
            )
    else:
        agg_ablation = abl_display_names = None

    # -------------------------------------------------------------------------
    # Save and plot — importance methods
    # -------------------------------------------------------------------------
    print("\nSaving results...")

    prefix = model_name

    # Permutation: aggregated bar chart + CSV
    if "permutation" in aggregated_methods:
        agg_perm = aggregated_methods["permutation"]
        plot_feature_importance(
            agg_perm,
            top_k=top_k,
            save_path=os.path.join(save_dir, f"{prefix}_kfold_importance.png"),
        )
        save_importance_results(
            agg_perm,
            os.path.join(save_dir, f"{prefix}_kfold_importance.csv"),
        )
        save_permutation_per_fold(
            fold_method_results, feature_cols, save_dir, prefix
        )
        save_stability(
            fold_method_results, feature_cols,
            aggregated_methods, save_dir, prefix,
        )

    # Multi-method side-by-side plot + CSVs
    if len(aggregated_methods) >= 1:
        plot_multi_method_importance(
            aggregated_methods,
            feature_cols,
            top_k=top_k,
            save_path=os.path.join(save_dir, f"{prefix}_all_methods.png"),
        )

    if corr_matrix is not None:
        plot_method_comparison(
            corr_matrix, pval_matrix, method_names,
            save_path=os.path.join(
                save_dir, f"{prefix}_method_corr_heatmap.png"
            ),
        )
        save_method_comparison(
            aggregated_methods, corr_matrix, pval_matrix,
            method_names, feature_cols, save_dir, prefix,
        )

    # -------------------------------------------------------------------------
    # Save and plot — ablation
    # -------------------------------------------------------------------------
    if agg_ablation is not None:
        plot_ablation_results(
            agg_ablation,
            abl_display_names,
            metric=metric,
            save_path=os.path.join(save_dir, f"{prefix}_ablation.png"),
        )
        save_ablation_results(
            agg_ablation,
            abl_display_names,
            feature_cols,
            group_indices,
            metric,
            save_dir,
            prefix,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {save_dir}/")
    print("\nKey output files:")
    print(f"  {prefix}_all_methods.png          — importance by all methods (side-by-side)")
    print(f"  {prefix}_method_corr_heatmap.png  — Spearman ρ heatmap")
    print(f"  {prefix}_method_correlations.csv  — Spearman ρ values")
    print(f"  {prefix}_kfold_importance.png/csv — permutation importance")
    print(f"  {prefix}_stability.csv            — CV stability scores")
    print(f"  {prefix}_ablation.png/csv         — channel group ablation")


if __name__ == "__main__":
    main()
