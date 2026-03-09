"""
Create stratified k-fold splits and save to a folds_meta.json file.

This script is the single source of truth for fold assignments.  Both
train_kfold.py and analyze_features_kfold.py consume the JSON it produces,
guaranteeing that training and analysis always use exactly the same splits.

Usage
-----
    python scripts/data_utils/create_folds.py \\
        --config examples/configs/analyze_kfold.yaml \\
        --output folds_meta.json
"""

import sys
import os
import json
import argparse
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold

from scripts.dataloaders.dataloader import list_csvs_by_class, get_file_labels
from scripts.utils.utils import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified k-fold splits and save metadata to JSON"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (same one used for training/analysis)",
    )
    parser.add_argument(
        "--output", default="folds_meta.json",
        help="Output path for the folds metadata JSON (default: folds_meta.json)",
    )
    args = parser.parse_args()

    cfg       = load_config(args.config)
    data_cfg  = cfg["data"]
    kfold_cfg = cfg["kfold"]

    n_folds   = kfold_cfg["n_folds"]
    seed      = data_cfg.get("seed", 42)
    data_root = data_cfg["root"]
    label_map = {"CONTROL": 0, "POSITIVE": 1}

    set_seed(seed)

    print("=" * 70)
    print("Creating K-Fold Splits")
    print("=" * 70)
    print(f"  Data root : {data_root}")
    print(f"  K-folds   : {n_folds}")
    print(f"  Seed      : {seed}")
    print(f"  Output    : {args.output}\n")

    all_files = list_csvs_by_class(data_root)
    labels    = get_file_labels(all_files, label_map)

    inv_label_map = {v: k for k, v in label_map.items()}
    overall_counts = {
        inv_label_map[k]: int(v) for k, v in Counter(labels).items()
    }
    print(f"Total files : {len(all_files)}  {overall_counts}\n")

    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    print(f"  {'Fold':<6}  {'Train':>6}  {'Test':>6}  Train class split  Test class split")
    print(f"  {'-'*65}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_files, labels)):
        train_files  = [all_files[i] for i in train_idx]
        test_files   = [all_files[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels  = labels[test_idx]

        train_counts = {inv_label_map[k]: int(v) for k, v in Counter(train_labels).items()}
        test_counts  = {inv_label_map[k]: int(v) for k, v in Counter(test_labels).items()}

        print(
            f"  {fold_idx+1:<6}  {len(train_files):>6}  {len(test_files):>6}"
            f"  {train_counts}  {test_counts}"
        )

        folds.append({
            "fold": fold_idx + 1,
            "train": {
                "files":  train_files,
                "counts": train_counts,
                "total":  len(train_files),
            },
            "test": {
                "files":  test_files,
                "counts": test_counts,
                "total":  len(test_files),
            },
        })

    # Aggregate summary stats across folds
    classes = list(label_map.keys())
    summary: dict = {}
    for split in ("train", "test"):
        summary[split] = {}
        for cls in classes:
            counts = [f[split]["counts"].get(cls, 0) for f in folds]
            summary[split][cls] = {
                "mean": round(float(np.mean(counts)), 2),
                "std":  round(float(np.std(counts)),  2),
                "min":  int(min(counts)),
                "max":  int(max(counts)),
            }

    print(f"\n  Summary (mean ± std across folds):")
    for split in ("train", "test"):
        parts = "  ".join(
            f"{cls}: {summary[split][cls]['mean']:.1f}±{summary[split][cls]['std']:.1f}"
            for cls in classes
        )
        print(f"    {split.capitalize():6s}: {parts}")

    meta = {
        "created":     datetime.now().isoformat(),
        "config_file": os.path.abspath(args.config),
        "config": {
            "data_root": data_root,
            "n_folds":   n_folds,
            "seed":      seed,
        },
        "label_map":     label_map,
        "total_files":   len(all_files),
        "overall_counts": overall_counts,
        "summary":       summary,
        "folds":         folds,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved fold metadata to: {args.output}")


if __name__ == "__main__":
    main()
