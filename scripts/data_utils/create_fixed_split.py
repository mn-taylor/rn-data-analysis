"""
Create a folds_meta.json from a pre-defined train/eval split.

Produces a single-"fold" folds_meta.json that is fully compatible with
train_kfold.py and analyze_features_kfold.py so those scripts work without
any modification.

Usage
-----
    python scripts/data_utils/create_fixed_split.py \\
        --train-dir data/may2026_parsed_data/train \\
        --eval-dir  data/may2026_parsed_data/eval \\
        --output    data/may2026_parsed_data/fixed_split_meta.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.dataloaders.dataloader import list_csvs_by_class, get_file_labels

LABEL_MAP = {"CONTROL": 0, "POSITIVE": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Create a fixed-split folds_meta.json from pre-defined train/eval dirs"
    )
    parser.add_argument(
        "--train-dir", required=True,
        help="Path to training data dir (must contain CONTROL/ and POSITIVE/ subdirs)",
    )
    parser.add_argument(
        "--eval-dir", required=True,
        help="Path to evaluation data dir (must contain CONTROL/ and POSITIVE/ subdirs)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the fixed_split_meta.json file",
    )
    args = parser.parse_args()

    train_files = list_csvs_by_class(args.train_dir)
    eval_files  = list_csvs_by_class(args.eval_dir)

    if not train_files:
        print(f"ERROR: No CSV files found in {args.train_dir}", file=sys.stderr)
        sys.exit(1)
    if not eval_files:
        print(f"ERROR: No CSV files found in {args.eval_dir}", file=sys.stderr)
        sys.exit(1)

    train_labels = get_file_labels(train_files, LABEL_MAP)
    eval_labels  = get_file_labels(eval_files,  LABEL_MAP)

    train_counts   = dict(Counter(INV_LABEL_MAP[l] for l in train_labels))
    eval_counts    = dict(Counter(INV_LABEL_MAP[l] for l in eval_labels))
    overall_counts = dict(Counter(
        INV_LABEL_MAP[l] for l in list(train_labels) + list(eval_labels)
    ))

    folds_meta = {
        "created":    datetime.now().isoformat(),
        "split_type": "fixed",
        "config": {
            "n_folds":   1,
            "train_dir": args.train_dir,
            "eval_dir":  args.eval_dir,
        },
        "label_map":      LABEL_MAP,
        "total_files":    len(train_files) + len(eval_files),
        "overall_counts": overall_counts,
        "folds": [
            {
                "fold":  1,
                "train": {"files": train_files, "counts": train_counts, "total": len(train_files)},
                "test":  {"files": eval_files,  "counts": eval_counts,  "total": len(eval_files)},
            }
        ],
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(folds_meta, f, indent=2)

    print(f"Saved fixed split meta to: {args.output}")
    print(f"  Train : {len(train_files)} files  {train_counts}")
    print(f"  Eval  : {len(eval_files)} files  {eval_counts}")
    print(f"  Total : {len(train_files) + len(eval_files)} files  {overall_counts}")


if __name__ == "__main__":
    main()
