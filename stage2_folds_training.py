"""
Stage 2: Monopoly (energy-based) training — per fold.

Assumes:
- Stage-1 models live in: models/fold_{k}/<label>_absolute.pth
- Folds defined in: splits/folds.json

For fold k:
- Test set = folds[k][label]
- Train set = all remaining CSVs for that label

If exactly two labels are present, computes ROC AUC.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# =========================
# Config
# =========================

@dataclass
class Stage2Config:
    data_dir: str = "data"
    splits_dir: str = "splits"
    models_dir: str = "models"

    fold_idx: int = 0

    delta: int = 25
    max_time: float = 600.0

    epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 1

    exclude_labels: List[str] = None
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_folds(splits_dir: str) -> Dict:
    path = os.path.join(splits_dir, "folds.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing folds.json at {path}. Run make_folds.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


def make_step_pairs(df, feature_cols, delta, max_time):
    X, Y = [], []
    arr = df[feature_cols].to_numpy()

    for i in range(len(arr)):
        if i + delta >= len(arr):
            break
        t_norm = i / max_time
        X.append([t_norm] + arr[i].tolist())
        Y.append(arr[i + delta].tolist())

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )


def apply_standardization(X, meta):
    mean = torch.tensor(meta["mean"], device=X.device, dtype=X.dtype)
    std = torch.tensor(meta["std"], device=X.device, dtype=X.dtype)
    return (X - mean) / std


# =========================
# Dataset
# =========================

class TimeSeriesDataset(Dataset):
    """
    One CSV file = one classification sample
    """

    def __init__(self, samples, feature_cols, delta, max_time):
        self.samples = samples
        self.feature_cols = feature_cols
        self.delta = delta
        self.max_time = max_time

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        df = pd.read_csv(path)

        X, Y = make_step_pairs(
            df, self.feature_cols, self.delta, self.max_time
        )

        return X, Y, torch.tensor(label_idx)


# =========================
# Models
# =========================

class DynamicsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - 1),
        )

    def forward(self, x):
        return self.net(x)


def load_stage1_models(cfg: Stage2Config):
    fold_models_dir = os.path.join(cfg.models_dir, f"fold_{cfg.fold_idx}")
    if not os.path.isdir(fold_models_dir):
        raise FileNotFoundError(
            f"Missing Stage-1 models for fold {cfg.fold_idx}: {fold_models_dir}"
        )

    models = []
    labels = []
    excluded = set(cfg.exclude_labels or [])

    for fname in sorted(os.listdir(fold_models_dir)):
        if not fname.endswith("_absolute.pth"):
            continue

        label = fname.replace("_absolute.pth", "")
        if label in excluded:
            continue

        ckpt = torch.load(
            os.path.join(fold_models_dir, fname),
            map_location=cfg.device,
        )

        feature_cols = ckpt["feature_cols"]
        model = DynamicsModel(input_dim=1 + len(feature_cols)).to(cfg.device)
        model.load_state_dict(ckpt["model_state"])
        model.train()

        model.input_meta = ckpt["input_meta"]
        model.output_meta = ckpt["output_meta"]

        models.append(model)
        labels.append(label)

    if not models:
        raise RuntimeError("No Stage-1 models loaded.")

    return models, labels, feature_cols


# =========================
# Energy Classifier
# =========================

class EnergyClassifier(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, X, Y):
        energies = []

        for model in self.models:
            Xf = apply_standardization(X[:, 1:], model.input_meta)
            X_in = torch.cat([X[:, :1], Xf], dim=1)

            Y_std = apply_standardization(Y, model.output_meta)
            pred = model(X_in)

            mse = ((pred - Y_std) ** 2).mean()
            energies.append(mse)

        return -torch.stack(energies)  # logits


# =========================
# Training
# =========================

def train_stage2(cfg: Stage2Config):
    set_seed(cfg.seed)

    folds_data = load_folds(cfg.splits_dir)
    folds = folds_data["folds"]
    labels_all = folds_data["labels"]

    if cfg.fold_idx < 0 or cfg.fold_idx >= len(folds):
        raise ValueError("Invalid fold_idx")

    fold_test = folds[cfg.fold_idx]

    # ---- Load Stage-1 models ----
    models, labels, feature_cols = load_stage1_models(cfg)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    is_binary = (len(labels) == 2)
    if is_binary:
        print(f"Binary classification: {labels[0]} vs {labels[1]}")

    classifier = EnergyClassifier(models).to(cfg.device)

    # ---- Build datasets ----
    train_samples, test_samples = [], []

    for label in labels:
        label_dir = os.path.join(cfg.data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        all_csvs = sorted(
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.endswith(".csv")
        )

        test_csvs = set(fold_test.get(label, []))
        for p in all_csvs:
            if p in test_csvs:
                test_samples.append((p, label_to_idx[label]))
            else:
                train_samples.append((p, label_to_idx[label]))

    train_loader = DataLoader(
        TimeSeriesDataset(train_samples, feature_cols, cfg.delta, cfg.max_time),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        TimeSeriesDataset(test_samples, feature_cols, cfg.delta, cfg.max_time),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    test_aucs = []

    # ---- Training loop ----
    for epoch in range(cfg.epochs):
        classifier.train()
        correct, total = 0, 0

        for X, Y, y in train_loader:
            X, Y, y = X[0].to(cfg.device), Y[0].to(cfg.device), y.to(cfg.device)

            optimizer.zero_grad()
            logits = classifier(X, Y)
            loss = criterion(logits.unsqueeze(0), y)
            loss.backward()
            optimizer.step()

            correct += int(logits.argmax() == y.item())
            total += 1

        train_accs.append(correct / total)

        # ---- Test ----
        classifier.eval()
        correct, total = 0, 0
        auc_scores, auc_targets = [], []

        with torch.no_grad():
            for X, Y, y in test_loader:
                X, Y, y = X[0].to(cfg.device), Y[0].to(cfg.device), y.to(cfg.device)
                logits = classifier(X, Y)

                correct += int(logits.argmax() == y.item())
                total += 1

                if is_binary:
                    score = (logits[1] - logits[0]).item()
                    auc_scores.append(score)
                    auc_targets.append(y.item())

        test_acc = correct / max(total, 1)
        test_accs.append(test_acc)

        if is_binary and len(set(auc_targets)) == 2:
            auc = roc_auc_score(auc_targets, auc_scores)
            test_aucs.append(auc)
            print(
                f"Epoch {epoch+1:3d}/{cfg.epochs} | "
                f"Train Acc: {train_accs[-1]:.3f} | "
                f"Test Acc: {test_acc:.3f} | "
                f"AUC: {auc:.3f}"
            )
        else:
            print(
                f"Epoch {epoch+1:3d}/{cfg.epochs} | "
                f"Train Acc: {train_accs[-1]:.3f} | "
                f"Test Acc: {test_acc:.3f}"
            )

    # ---- Save outputs ----
    out_dir = os.path.join(cfg.models_dir, f"fold_{cfg.fold_idx}")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_accs, label="Train")
    plt.plot(test_accs, label="Test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Stage-2 Accuracy (Fold {cfg.fold_idx})")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "monopoly_accuracy.png"), dpi=150)
    plt.close()

    if is_binary and test_aucs:
        plt.figure()
        plt.plot(test_aucs)
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"Stage-2 AUC (Fold {cfg.fold_idx})")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, "monopoly_auc.png"), dpi=150)
        plt.close()

    print(f"Finished Stage-2 for fold {cfg.fold_idx}")
    return auc


# =========================
# Main
# =========================

if __name__ == "__main__":
    aucs = []
    for fold in range(5):
        cfg = Stage2Config(
            data_dir="data",
            splits_dir="splits",
            models_dir="models_folds",
            fold_idx=fold,
            epochs=20,
            exclude_labels=["PSA"],
        )

        aucs.append(train_stage2(cfg))

    print(aucs)
