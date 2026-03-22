"""
K-fold training — config-driven entry point.

Trains one model checkpoint per fold and evaluates it on the held-out test
set, reporting all classification metrics.  Fold assignments are loaded from
a pre-generated folds_meta.json (created by create_folds.py) so that training
and analysis always use exactly the same splits.

Usage
-----
    python examples/train_kfold.py \\
        --config  examples/configs/analyze_kfold.yaml \\
        --folds-meta folds_meta.json

Outputs (written to output.save_dir)
-------------------------------------
  {model}_fold{k}.pth          — per-fold model checkpoint
  {model}_train_metrics.csv    — per-fold test metrics table
  {model}_train_summary.json   — full results summary (JSON)
  {model}_confusion_matrix.png — aggregated confusion matrix
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import pandas as pd
import numpy as np

from scripts.utils.config import DataConfig
from scripts.dataloaders.dataloader import create_dataloaders
from scripts.models import (
    ImprovedCNN1D,
    ResNet1D,
    InceptionTime,
    RocketClassifier,
    TimesNetWrapper,
    TSLANetWrapper,
)
from scripts.train import train_model, compute_full_metrics
from scripts.utils.utils import set_seed, get_device, plot_confusion_matrix


# =============================================================================
# Helpers
# =============================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_folds_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_model(model_name: str, input_channels: int, model_cfg: dict, seq_len: int = 512):
    import types
    dropout = model_cfg.get("dropout", 0.3)
    if model_name == "ImprovedCNN1D":
        return ImprovedCNN1D(C=input_channels, dropout=dropout)
    elif model_name == "ResNet1D":
        return ResNet1D(input_channels=input_channels, num_classes=2, dropout=dropout)
    elif model_name == "InceptionTime":
        return InceptionTime(
            input_channels=input_channels, num_classes=2,
            n_filters=model_cfg.get("n_filters", 32),
            depth=model_cfg.get("depth", 6),
            dropout=dropout,
        )
    elif model_name == "ROCKET":
        return RocketClassifier(
            input_channels=input_channels, num_classes=2,
            num_kernels=model_cfg.get("num_kernels", 5000),
            dropout=dropout,
        )
    elif model_name == "TimesNet":
        configs = types.SimpleNamespace(
            task_name   = "classification",
            seq_len     = seq_len,
            label_len   = 0,
            pred_len    = 0,
            enc_in      = input_channels,
            num_class   = 2,
            d_model     = model_cfg.get("d_model", 64),
            d_ff        = model_cfg.get("d_ff", 64),
            e_layers    = model_cfg.get("e_layers", 2),
            embed       = model_cfg.get("embed", "timeF"),
            freq        = model_cfg.get("freq", "h"),
            dropout     = dropout,
            top_k       = model_cfg.get("top_k", 3),
            num_kernels = model_cfg.get("num_kernels", 3),
        )
        return TimesNetWrapper(configs)
    elif model_name == "TSLANet":
        configs = types.SimpleNamespace(
            seq_len         = seq_len,
            enc_in          = input_channels,
            num_class       = 2,
            emb_dim         = model_cfg.get("emb_dim", 128),
            depth           = model_cfg.get("depth", 2),
            patch_size      = model_cfg.get("patch_size", 8),
            dropout         = model_cfg.get("dropout", 0.15),
            use_asb         = model_cfg.get("use_asb", True),
            use_icb         = model_cfg.get("use_icb", True),
            adaptive_filter = model_cfg.get("adaptive_filter", True),
        )
        return TSLANetWrapper(configs)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="K-fold training (config-driven)")
    parser.add_argument(
        "--config", type=str, default="examples/configs/analyze_kfold.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--folds-meta", type=str, required=True,
        help="Path to folds_meta.json produced by create_folds.py",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name from config (ImprovedCNN1D | ResNet1D | InceptionTime | ROCKET | TimesNet | TSLANet)",
    )
    parser.add_argument(
        "--run-save-dir", type=str, default=None,
        help="Explicit output directory for this run. "
             "If omitted, auto-generates results_root/{model}/{model}_run_{timestamp}/",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data.root from config (path containing POSITIVE/ and CONTROL/ subdirs)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device from config (e.g. cuda:0, cuda:1, cpu)",
    )
    args = parser.parse_args()

    cfg        = load_config(args.config)
    folds_meta = load_folds_meta(args.folds_meta)

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    output_cfg   = cfg["output"]

    # CLI overrides
    if args.data_root is not None:
        data_cfg = dict(data_cfg)
        data_cfg["root"] = args.data_root
    if args.model is not None:
        model_cfg = dict(model_cfg)
        model_cfg["name"] = args.model

    model_name = model_cfg["name"]
    n_folds    = folds_meta["config"]["n_folds"]

    # Resolve output directory
    if args.run_save_dir is not None:
        save_dir = args.run_save_dir
    else:
        results_root = output_cfg.get("results_root", "results")
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir     = os.path.join(results_root, model_name, f"{model_name}_run_{timestamp}")

    print("=" * 80)
    print("K-Fold Training")
    print("=" * 80)
    print(f"\n  Model      : {model_name}")
    print(f"  K-folds    : {n_folds}")
    print(f"  Folds meta : {args.folds_meta}")
    print(f"  Data root  : {data_cfg['root']}")
    print(f"  Run dir    : {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    set_seed(data_cfg.get("seed", 42))
    if args.device is not None:
        cfg["device"] = args.device
    device = get_device(cfg.get("device", "cuda"))
    print(f"\n  Device: {device}")

    # Detect input channels from first file in fold 0
    first_file = folds_meta["folds"][0]["train"]["files"][0]
    df0 = pd.read_csv(first_file)
    id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
    feature_cols   = [c for c in df0.columns if c not in id_cols]
    input_channels = len(feature_cols)
    print(f"  Input channels: {input_channels}")

    label_map   = folds_meta["label_map"]
    data_config = DataConfig(
        root=data_cfg["root"],
        T=data_cfg.get("T", 512),
        batch_size=data_cfg.get("batch_size", 32),
        seed=data_cfg.get("seed", 42),
        label_map=label_map,
    )

    # -------------------------------------------------------------------------
    # K-Fold training loop
    # -------------------------------------------------------------------------
    fold_metrics_list = []
    _metric_keys = ["auc", "accuracy", "sensitivity", "specificity", "ppv", "brier_score"]

    for fold_entry in folds_meta["folds"]:
        fold_idx    = fold_entry["fold"] - 1
        train_files = fold_entry["train"]["files"]
        test_files  = fold_entry["test"]["files"]

        print(f"\n{'-' * 80}")
        print(f"FOLD {fold_idx + 1} / {n_folds}")
        print(f"{'-' * 80}")
        print(f"  Train: {len(train_files)}  {fold_entry['train']['counts']}")
        print(f"  Test : {len(test_files)}   {fold_entry['test']['counts']}")

        train_loader, test_loader = create_dataloaders(
            train_files, test_files,
            label_map=data_config.label_map,
            T=data_config.T,
            batch_size=data_config.batch_size,
            output_format="channels_first",
        )

        model_path = os.path.join(save_dir, f"{model_name}_fold{fold_idx}.pth")
        model      = build_model(model_name, input_channels, model_cfg, seq_len=data_config.T)

        if os.path.exists(model_path):
            print(f"  Loading existing checkpoint: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"  Training from scratch...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_cfg.get("lr", 1e-3),
                weight_decay=training_cfg.get("weight_decay", 1e-2),
            )
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                loss_fn=torch.nn.CrossEntropyLoss(),
                device=device,
                epochs=training_cfg.get("epochs", 50),
                early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
                checkpoint_path=model_path,
            )
            print(f"  Saved checkpoint: {model_path}")

        model = model.to(device)
        model.eval()

        metrics = compute_full_metrics(model, test_loader, device)
        fold_metrics_list.append(metrics)

        print(
            f"\n  AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
            f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
            f"PPV={metrics['ppv']:.4f}  Brier={metrics['brier_score']:.4f}"
        )
        print(
            f"  TP={metrics['tp']}  TN={metrics['tn']}  "
            f"FP={metrics['fp']}  FN={metrics['fn']}"
        )

    # -------------------------------------------------------------------------
    # Aggregate and print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"\n  {'Metric':<14}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*34}")
    metric_labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "Brier Score"]
    for key, label in zip(_metric_keys, metric_labels):
        vals = [m[key] for m in fold_metrics_list]
        print(f"  {label:<14}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}")

    # -------------------------------------------------------------------------
    # Save per-fold metrics CSV
    # -------------------------------------------------------------------------
    rows = []
    for fold_idx, m in enumerate(fold_metrics_list):
        row = {"fold": fold_idx + 1}
        for key in _metric_keys:
            row[key] = m[key]
        row.update({"tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"]})
        rows.append(row)

    summary_row = {"fold": "mean±std"}
    for key in _metric_keys:
        vals = [m[key] for m in fold_metrics_list]
        summary_row[key] = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
    rows.append(summary_row)

    csv_path = os.path.join(save_dir, f"{model_name}_train_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved metrics table  : {csv_path}")

    # -------------------------------------------------------------------------
    # Save results JSON
    # -------------------------------------------------------------------------
    summed_cm = sum(m["confusion_matrix"] for m in fold_metrics_list)
    results_summary = {
        "model":      model_name,
        "n_folds":    n_folds,
        "folds_meta": args.folds_meta,
        "metric_summary": {
            key: {
                "mean":     float(np.mean([m[key] for m in fold_metrics_list])),
                "std":      float(np.std( [m[key] for m in fold_metrics_list])),
                "per_fold": [float(m[key]) for m in fold_metrics_list],
            }
            for key in _metric_keys
        },
        "confusion_matrix_sum": summed_cm.tolist(),
    }
    json_path = os.path.join(save_dir, f"{model_name}_train_summary.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Saved results summary: {json_path}")

    # -------------------------------------------------------------------------
    # Confusion matrix plot
    # -------------------------------------------------------------------------
    plot_confusion_matrix(
        summed_cm,
        title=f"Confusion Matrix — {n_folds}-Fold ({model_name})",
        save_path=os.path.join(save_dir, f"{model_name}_confusion_matrix.png"),
    )

    print(f"\nAll outputs saved to: {save_dir}/")


if __name__ == "__main__":
    main()
