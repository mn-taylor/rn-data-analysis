"""
Stage 1: Independent class-conditional predictor training (per fold).

Assumes dataset layout:

data/
  class_a/
    sample1.csv
    sample2.csv
  class_b/
    sample3.csv

Splits are provided externally via:
  splits/folds.json

For fold k:
- Test set = folds[k][label]
- Train set = all other CSVs for that label

Outputs:
  models/fold_k/<label>_absolute.pth
  models/fold_k/<label>_absolute_loss.png
"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# Config
# =========================

@dataclass
class Stage1Config:
    data_dir: str = "data"
    splits_dir: str = "splits"
    models_dir: str = "models"

    fold_idx: int = 0

    feature_cols: List[str] = None
    delta: int = 25
    max_time: float = 600.0

    epochs: int = 200
    lr: float = 1e-4
    batch_size: int = 128

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
            f"Missing folds.json at {path}. "
            "Run make_folds.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


def standardize(X: torch.Tensor, eps: float = 1e-8):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    Z = (X - mean) / std
    meta = {"mean": mean.cpu().numpy(), "std": std.cpu().numpy()}
    return Z, meta


def apply_standardization(X: torch.Tensor, meta: Dict):
    mean = torch.tensor(meta["mean"], device=X.device, dtype=X.dtype)
    std = torch.tensor(meta["std"], device=X.device, dtype=X.dtype)
    return (X - mean) / std


# =========================
# Dataset construction
# =========================

def make_step_pairs(
    df: pd.DataFrame,
    feature_cols: List[str],
    delta: int,
    max_time: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    arr = df[feature_cols].to_numpy()
    X, Y = [], []

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


class NextStepDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =========================
# Model
# =========================

class DynamicsModel(nn.Module):
    """
    [t_norm, x_t] -> x_{t+Δ}
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
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


# =========================
# Training (single class)
# =========================

def train_single_class(
    label: str,
    train_csvs: List[str],
    test_csvs: List[str],
    cfg: Stage1Config,
    out_dir: str,
):
    print(f"\n=== Training class: {label} ===")
    print(f"Train CSVs: {len(train_csvs)} | Test CSVs: {len(test_csvs)}")

    train_dfs = [pd.read_csv(p) for p in train_csvs]
    test_dfs = [pd.read_csv(p) for p in test_csvs]

    # ---- Infer feature columns ----
    if cfg.feature_cols is None:
        cfg.feature_cols = [
            c for c in train_dfs[0].columns
            if np.issubdtype(train_dfs[0][c].dtype, np.number)
        ]

    # ---- Build train data ----
    X_tr, Y_tr = [], []
    for df in train_dfs:
        X, Y = make_step_pairs(df, cfg.feature_cols, cfg.delta, cfg.max_time)
        if len(X) > 0:
            X_tr.append(X)
            Y_tr.append(Y)

    if not X_tr:
        raise RuntimeError(f"No valid training samples for class '{label}'")

    X_tr = torch.cat(X_tr, dim=0)
    Y_tr = torch.cat(Y_tr, dim=0)

    # ---- Build test data ----
    X_te, Y_te = [], []
    for df in test_dfs:
        X, Y = make_step_pairs(df, cfg.feature_cols, cfg.delta, cfg.max_time)
        if len(X) > 0:
            X_te.append(X)
            Y_te.append(Y)

    X_te = torch.cat(X_te, dim=0)
    Y_te = torch.cat(Y_te, dim=0)

    # ---- Standardization (train only) ----
    Y_tr_std, output_meta = standardize(Y_tr)
    X_tr_feat_std, input_meta = standardize(X_tr[:, 1:])

    X_tr[:, 1:] = X_tr_feat_std
    X_te[:, 1:] = apply_standardization(X_te[:, 1:], input_meta)
    Y_te_std = apply_standardization(Y_te, output_meta)

    train_loader = DataLoader(
        NextStepDataset(X_tr, Y_tr_std),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        NextStepDataset(X_te, Y_te_std),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = DynamicsModel(input_dim=X_tr.shape[1]).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []

    # ---- Training loop ----
    for epoch in tqdm(range(cfg.epochs), desc=label):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                test_loss += loss_fn(model(xb), yb).item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

    # ---- Save plot ----
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"{label} — Stage 1 (Fold {cfg.fold_idx})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"{label}_absolute_loss.png"), dpi=150)
    plt.close()

    # ---- Save model ----
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_meta": input_meta,
            "output_meta": output_meta,
            "feature_cols": cfg.feature_cols,
        },
        os.path.join(out_dir, f"{label}_absolute.pth"),
    )

    print(f"Saved model for '{label}'")


# =========================
# Stage-1 entry point
# =========================

def train_stage1(cfg: Stage1Config):
    set_seed(cfg.seed)

    folds_data = load_folds(cfg.splits_dir)
    folds = folds_data["folds"]
    labels = folds_data["labels"]

    if cfg.fold_idx < 0 or cfg.fold_idx >= len(folds):
        raise ValueError(
            f"fold_idx={cfg.fold_idx} invalid for {len(folds)} folds"
        )

    fold_test = folds[cfg.fold_idx]

    print(
        f"\n=== Stage-1 Training | Fold {cfg.fold_idx} / {len(folds)-1} ==="
    )

    fold_models_dir = os.path.join(cfg.models_dir, f"fold_{cfg.fold_idx}")
    os.makedirs(fold_models_dir, exist_ok=True)

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
        train_csvs = [p for p in all_csvs if p not in test_csvs]

        if not train_csvs:
            raise RuntimeError(
                f"No training CSVs for class '{label}' in fold {cfg.fold_idx}"
            )

        train_single_class(
            label=label,
            train_csvs=train_csvs,
            test_csvs=list(test_csvs),
            cfg=cfg,
            out_dir=fold_models_dir,
        )


# =========================
# Main
# =========================

if __name__ == "__main__":
    for fold in range(0, 5):
        cfg = Stage1Config(
            data_dir="data",
            splits_dir="splits",
            models_dir="models_folds",
            fold_idx=fold,
            delta=25,
            epochs=200,
            lr=1e-4,
            batch_size=128,
        )

        train_stage1(cfg)
