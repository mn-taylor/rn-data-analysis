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
  {model}_test_metrics.csv            — per-fold AUC/Acc/Sens/Spec/PPV/Brier
  {model}_confusion_matrix.png        — aggregated confusion matrix
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import torch
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from scripts.utils.config import DataConfig
from scripts.dataloaders.dataloader import (
    create_dataloaders,
    list_csvs_by_class,
    get_file_labels,
)
from scripts.models import (
    ImprovedCNN1D,
    ResNet1D,
    InceptionTime,
    RocketClassifier,
)
from scripts.train import train_model, compute_full_metrics
from scripts.utils.utils import set_seed, get_device, plot_confusion_matrix
from scripts.interpretability.feature_importance import (
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
from scripts.interpretability.channel_ablation import (
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

def load_folds_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


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
    parser.add_argument(
        "--folds-meta",
        type=str,
        default=None,
        help="Path to folds_meta.json produced by create_folds.py. "
             "If omitted, folds are created on the fly from the data root in the config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config (ImprovedCNN1D | ResNet1D | InceptionTime | ROCKET)",
    )
    parser.add_argument(
        "--run-save-dir",
        type=str,
        default=None,
        help="Explicit output directory for this run (checkpoints + analysis outputs). "
             "If omitted, auto-generates results_root/{model}/{model}_run_{timestamp}/",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data.root from config (path containing POSITIVE/ and CONTROL/ subdirs)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply data-root override before anything reads cfg["data"]["root"]
    if args.data_root is not None:
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["root"] = args.data_root

    # -------------------------------------------------------------------------
    # Load or create fold splits
    # -------------------------------------------------------------------------
    if args.folds_meta is not None:
        print(f"\nLoading fold splits from: {args.folds_meta}")
        folds_meta = load_folds_meta(args.folds_meta)
        print(f"  Loaded {folds_meta['config']['n_folds']}-fold splits "
              f"(created: {folds_meta.get('created', 'unknown')})")
    else:
        print("\nNo folds meta provided — creating folds on the fly...")
        data_cfg_tmp  = cfg["data"]
        kfold_cfg_tmp = cfg["kfold"]
        n_folds_tmp   = kfold_cfg_tmp["n_folds"]
        seed_tmp      = data_cfg_tmp.get("seed", 42)
        label_map_tmp = {"CONTROL": 0, "POSITIVE": 1}
        inv_label_map = {v: k for k, v in label_map_tmp.items()}

        all_files_tmp = list_csvs_by_class(data_cfg_tmp["root"])
        labels_tmp    = get_file_labels(all_files_tmp, label_map_tmp)
        print(f"  Found {len(all_files_tmp)} files  "
              f"{dict(Counter(inv_label_map[l] for l in labels_tmp))}")

        skf_tmp = StratifiedKFold(n_splits=n_folds_tmp, shuffle=True, random_state=seed_tmp)
        folds_tmp = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf_tmp.split(all_files_tmp, labels_tmp)):
            train_files_t  = [all_files_tmp[i] for i in train_idx]
            test_files_t   = [all_files_tmp[i] for i in test_idx]
            train_counts_t = dict(Counter(inv_label_map[labels_tmp[i]] for i in train_idx))
            test_counts_t  = dict(Counter(inv_label_map[labels_tmp[i]] for i in test_idx))
            folds_tmp.append({
                "fold":  fold_idx + 1,
                "train": {"files": train_files_t, "counts": train_counts_t, "total": len(train_files_t)},
                "test":  {"files": test_files_t,  "counts": test_counts_t,  "total": len(test_files_t)},
            })

        folds_meta = {
            "created": "on-the-fly",
            "config":  {"n_folds": n_folds_tmp, "seed": seed_tmp},
            "label_map":      label_map_tmp,
            "total_files":    len(all_files_tmp),
            "overall_counts": dict(Counter(inv_label_map[l] for l in labels_tmp)),
            "folds":          folds_tmp,
        }
        print(f"  Created {n_folds_tmp} folds on the fly (seed={seed_tmp})")

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    analysis_cfg = cfg["analysis"]
    groups_cfg   = cfg.get("channel_groups", {})
    output_cfg   = cfg["output"]

    # CLI --model overrides the config value
    if args.model is not None:
        model_cfg = dict(model_cfg)
        model_cfg["name"] = args.model

    model_name = model_cfg["name"]
    n_folds    = folds_meta["config"]["n_folds"]
    top_k      = output_cfg.get("top_k", 20)
    metric     = analysis_cfg.get("metric", "accuracy")

    # Resolve output directory (checkpoints + all analysis outputs live here)
    if args.run_save_dir is not None:
        save_dir = args.run_save_dir
    else:
        results_root = output_cfg.get("results_root", "results")
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir     = os.path.join(results_root, model_name, f"{model_name}_run_{timestamp}")

    print("=" * 80)
    print("K-Fold Feature Importance Analysis")
    print("=" * 80)
    print(f"\n  Model      : {model_name}")
    print(f"  K-folds    : {n_folds}")
    print(f"  Folds meta : {args.folds_meta or '(on the fly)'}")
    print(f"  Data root  : {data_cfg['root']}")
    print(f"  Methods    : {analysis_cfg.get('methods', [])}")
    print(f"  Run dir    : {save_dir}")

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

    # Detect feature columns from first train file in fold 0
    first_file = folds_meta["folds"][0]["train"]["files"][0]
    df0 = pd.read_csv(first_file)
    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
    feature_cols   = [c for c in df0.columns if c not in id_cols]
    input_channels = len(feature_cols)
    print(f"  Input channels: {input_channels}")
    print(f"  Total files   : {folds_meta['total_files']}")

    # Identify channel groups for ablation
    print("\nChannel groups:")
    group_indices = identify_channel_groups(feature_cols, groups_cfg)

    # -------------------------------------------------------------------------
    # K-Fold loop
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print(f"# K-Fold Analysis ({n_folds} folds)")
    print(f"{'#' * 80}")

    fold_method_results: list = []   # list of method->result dicts
    fold_ablation_results: list = [] # list of (results, display_names) tuples
    fold_metrics_list: list = []     # list of compute_full_metrics dicts

    for fold_entry in folds_meta["folds"]:
        fold_idx   = fold_entry["fold"] - 1
        train_files = fold_entry["train"]["files"]
        test_files  = fold_entry["test"]["files"]

        print(f"\n{'-' * 80}")
        print(f"FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'-' * 80}")

        train_counts = fold_entry["train"]["counts"]
        test_counts  = fold_entry["test"]["counts"]
        train_total  = fold_entry["train"]["total"]
        test_total   = fold_entry["test"]["total"]

        # Class counts + ratios
        train_ratio_str = "  ".join(
            f"{cls}: {cnt}/{train_total} ({cnt/train_total:.1%})"
            for cls, cnt in sorted(train_counts.items())
        )
        test_ratio_str = "  ".join(
            f"{cls}: {cnt}/{test_total} ({cnt/test_total:.1%})"
            for cls, cnt in sorted(test_counts.items())
        )
        print(f"  Train ({train_total} files): {train_ratio_str}")
        print(f"  Test  ({test_total} files):  {test_ratio_str}")

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

        # -- Full test-set metrics --
        print(f"\n  Computing test metrics for fold {fold_idx + 1}...")
        fold_metrics = compute_full_metrics(model, test_loader, device)
        fold_metrics_list.append(fold_metrics)
        print(
            f"  AUC={fold_metrics['auc']:.4f}  "
            f"Acc={fold_metrics['accuracy']:.4f}  "
            f"Sens={fold_metrics['sensitivity']:.4f}  "
            f"Spec={fold_metrics['specificity']:.4f}  "
            f"PPV={fold_metrics['ppv']:.4f}  "
            f"Brier={fold_metrics['brier_score']:.4f}  "
            f"[TP={fold_metrics['tp']} TN={fold_metrics['tn']} "
            f"FP={fold_metrics['fp']} FN={fold_metrics['fn']}]"
        )

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
    # Aggregate test metrics across folds
    # -------------------------------------------------------------------------
    print(f"\n{'#' * 80}")
    print("# Test Metrics Summary")
    print(f"{'#' * 80}\n")

    _metric_keys = ["auc", "accuracy", "sensitivity", "specificity", "ppv", "brier_score"]
    _metric_labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "Brier Score"]

    print(f"  {'Metric':<14}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*34}")
    for key, label in zip(_metric_keys, _metric_labels):
        vals = [m[key] for m in fold_metrics_list]
        print(f"  {label:<14}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}")

    # Save per-fold metrics to CSV
    metrics_rows = []
    for fold_idx, m in enumerate(fold_metrics_list):
        row = {"fold": fold_idx + 1}
        for key in _metric_keys:
            row[key] = m[key]
        row.update({"tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"]})
        metrics_rows.append(row)
    # Append summary row
    summary = {"fold": "mean±std"}
    for key in _metric_keys:
        vals = [m[key] for m in fold_metrics_list]
        summary[key] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    metrics_rows.append(summary)

    metrics_path = os.path.join(save_dir, f"{prefix}_test_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"\n  Saved test metrics to {metrics_path}")

    # Aggregated confusion matrix (sum across folds)
    summed_cm = sum(m["confusion_matrix"] for m in fold_metrics_list)
    plot_confusion_matrix(
        summed_cm,
        title=f"Confusion Matrix — {n_folds}-Fold Aggregate ({model_name})",
        save_path=os.path.join(save_dir, f"{prefix}_confusion_matrix.png"),
    )

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
    # Save classification results summary JSON
    # -------------------------------------------------------------------------
    _metric_keys = ["auc", "accuracy", "sensitivity", "specificity", "ppv", "brier_score"]
    results_summary = {
        "model":      model_name,
        "n_folds":    n_folds,
        "folds_meta": args.folds_meta,
        "metric_summary": {
            key: {
                "mean": float(np.mean([m[key] for m in fold_metrics_list])),
                "std":  float(np.std( [m[key] for m in fold_metrics_list])),
                "per_fold": [float(m[key]) for m in fold_metrics_list],
            }
            for key in _metric_keys
        },
        "confusion_matrix_sum": summed_cm.tolist(),
    }
    results_json_path = os.path.join(save_dir, f"{prefix}_results_summary.json")
    with open(results_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Saved results summary to {results_json_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {save_dir}/")
    print("\nKey output files:")
    print(f"  {prefix}_results_summary.json     — classification metrics (JSON)")
    print(f"  {prefix}_test_metrics.csv         — per-fold metrics table")
    print(f"  {prefix}_confusion_matrix.png     — aggregated confusion matrix")
    print(f"  {prefix}_all_methods.png          — importance by all methods (side-by-side)")
    print(f"  {prefix}_method_corr_heatmap.png  — Spearman ρ heatmap")
    print(f"  {prefix}_method_correlations.csv  — Spearman ρ values")
    print(f"  {prefix}_kfold_importance.png/csv — permutation importance")
    print(f"  {prefix}_stability.csv            — CV stability scores")
    print(f"  {prefix}_ablation.png/csv         — channel group ablation")


if __name__ == "__main__":
    main()
