"""
Generate file-level k-fold splits for time-series classification.

Assumes dataset layout:

data/
  class_a/
    sample1.csv
    sample2.csv
  class_b/
    sample3.csv

Each fold contains a per-class list of CSVs used as the TEST set.
Training set = all remaining files for that class.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List


# =========================
# Config
# =========================

@dataclass
class FoldConfig:
    data_dir: str = "data"
    splits_dir: str = "splits"
    k_folds: int = 5
    seed: int = 0


# =========================
# Fold generation
# =========================

def generate_k_folds(cfg: FoldConfig) -> Dict:
    """
    Generate stratified file-level k-fold splits.

    Returns dict with keys:
      - k_folds
      - labels
      - folds: List[Dict[label -> List[csv_paths]]]
    """

    rng = random.Random(cfg.seed)

    labels = sorted(
        d for d in os.listdir(cfg.data_dir)
        if os.path.isdir(os.path.join(cfg.data_dir, d))
    )

    if not labels:
        raise RuntimeError("No class folders found in data_dir.")

    # Initialize empty folds
    folds: List[Dict[str, List[str]]] = [
        {label: [] for label in labels}
        for _ in range(cfg.k_folds)
    ]

    for label in labels:
        label_dir = os.path.join(cfg.data_dir, label)
        csvs = sorted(
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.endswith(".csv")
        )

        if len(csvs) < cfg.k_folds:
            raise RuntimeError(
                f"Class '{label}' has {len(csvs)} files, "
                f"but k_folds={cfg.k_folds} was requested."
            )

        rng.shuffle(csvs)

        # Round-robin assignment
        for i, path in enumerate(csvs):
            fold_idx = i % cfg.k_folds
            folds[fold_idx][label].append(path)

    return {
        "k_folds": cfg.k_folds,
        "labels": labels,
        "folds": folds,
    }


# =========================
# Entry point
# =========================

def main():
    cfg = FoldConfig()

    os.makedirs(cfg.splits_dir, exist_ok=True)
    out_path = os.path.join(cfg.splits_dir, "folds.json")

    if os.path.exists(out_path):
        print(f"Folds already exist at {out_path}. Not overwriting.")
        return

    splits = generate_k_folds(cfg)

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(
        f"Saved {splits['k_folds']}-fold splits to {out_path}\n"
        f"Classes: {splits['labels']}"
    )


if __name__ == "__main__":
    main()
