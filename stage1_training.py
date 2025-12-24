"""
Stage 1: Independent class-conditional predictor training.

Assumes dataset layout:

data/
  class_a/
    sample1.csv
    sample2.csv
  class_b/
    sample3.csv

Each CSV:
- is time-ordered
- has numeric feature columns
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
    feature_cols: List[str] = None
    delta: int = 25
    max_time: float = 600.0

    epochs: int = 200
    lr: float = 1e-4
    batch_size: int = 128

    models_dir: str = "models"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Test split control ----
    n_test: int = 1                  # used if test_fraction is None
    test_fraction: float = None      # e.g. 0.2 for 20%
    splits_dir: str = "splits"



# =========================
# Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """
    Build (X, Y) where:
      X = [t_norm, features_t]
      Y = features_{t+delta}
    """
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
    Generic predictor:
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
# Training
# =========================
def build_or_load_test_split(
    data_dir: str,
    labels: List[str],
    n_test: int,
    test_fraction: float,
    splits_dir: str,
    seed: int,
) -> Dict[str, List[str]]:
    """
    Returns dict: label -> list of CSV paths held out for testing.

    Split logic:
    - If test_fraction is not None: use ceil(test_fraction * N)
    - Else: use n_test
    """
    os.makedirs(splits_dir, exist_ok=True)
    split_path = os.path.join(splits_dir, "test_set.json")

    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            print(f"Loaded existing test split from {split_path}")
            return json.load(f)

    rng = random.Random(seed)
    test_split = {}

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        csvs = sorted(
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.endswith(".csv")
        )

        if not csvs:
            raise RuntimeError(f"No CSV files found for class '{label}'")

        if test_fraction is not None:
            if not (0.0 < test_fraction < 1.0):
                raise ValueError("test_fraction must be in (0, 1)")
            k = max(1, int(round(test_fraction * len(csvs))))
        else:
            k = n_test

        if k >= len(csvs):
            raise RuntimeError(
                f"Class '{label}' has {len(csvs)} CSVs, "
                f"but requested {k} test samples."
            )

        test_csvs = rng.sample(csvs, k)
        test_split[label] = test_csvs

    with open(split_path, "w") as f:
        json.dump(test_split, f, indent=4)

    print(f"Saved test split to {split_path}")
    return test_split



def train_single_class(
    label: str,
    train_csvs: List[str],
    test_csvs: List[str],
    cfg: Stage1Config,
):
    print(f"\n=== Training class: {label} ===")

    train_dfs = [pd.read_csv(p) for p in train_csvs]
    test_dfs = [pd.read_csv(p) for p in test_csvs]

    # ---- Infer feature columns ----
    if cfg.feature_cols is None:
        cfg.feature_cols = [
            c for c in train_dfs[0].columns
            if np.issubdtype(train_dfs[0][c].dtype, np.number)
        ]

    # ---- Build train dataset ----
    X_tr, Y_tr = [], []
    for df in train_dfs:
        X, Y = make_step_pairs(
            df=df,
            feature_cols=cfg.feature_cols,
            delta=cfg.delta,
            max_time=cfg.max_time,
        )
        if len(X) > 0:
            X_tr.append(X)
            Y_tr.append(Y)

    if not X_tr:
        raise RuntimeError(f"No valid training samples for class '{label}'")

    X_tr = torch.cat(X_tr, dim=0)
    Y_tr = torch.cat(Y_tr, dim=0)

    # ---- Build test dataset ----
    X_te, Y_te = [], []
    for df in test_dfs:
        X, Y = make_step_pairs(
            df=df,
            feature_cols=cfg.feature_cols,
            delta=cfg.delta,
            max_time=cfg.max_time,
        )
        if len(X) > 0:
            X_te.append(X)
            Y_te.append(Y)

    X_te = torch.cat(X_te, dim=0)
    Y_te = torch.cat(Y_te, dim=0)

    # ---- Standardization (fit on train only) ----
    Y_tr_std, output_meta = standardize(Y_tr)
    X_tr_feat_std, input_meta = standardize(X_tr[:, 1:])

    X_tr[:, 1:] = X_tr_feat_std
    X_te[:, 1:] = apply_standardization(X_te[:, 1:], input_meta)
    Y_te_std = apply_standardization(Y_te, output_meta)

    # ---- Datasets ----
    train_dataset = NextStepDataset(X_tr, Y_tr_std)
    test_dataset = NextStepDataset(X_te, Y_te_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ---- Model ----
    model = DynamicsModel(input_dim=X_tr.shape[1]).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []

    # ---- Training loop ----
    for epoch in tqdm(range(cfg.epochs), desc=label):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---- Test loss ----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"[{label}] Epoch {epoch+1:4d}/{cfg.epochs} | "
                f"Train: {train_loss:.6f} | Test: {test_loss:.6f}"
            )

    # ---- Save loss plot ----
    os.makedirs(cfg.models_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{label} — Stage 1 Training")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = os.path.join(cfg.models_dir, f"{label}_absolute_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Save model ----
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_meta": input_meta,
            "output_meta": output_meta,
            "feature_cols": cfg.feature_cols,
        },
        os.path.join(cfg.models_dir, f"{label}_absolute.pth"),
    )

    print(f"Saved model + loss plot for '{label}'")




# =========================
# Entry point
# =========================

def train_stage1(cfg: Stage1Config):
    set_seed(cfg.seed)

    labels = [
        d for d in os.listdir(cfg.data_dir)
        if os.path.isdir(os.path.join(cfg.data_dir, d))
    ]

    test_split = build_or_load_test_split(
        data_dir=cfg.data_dir,
        labels=labels,
        n_test=cfg.n_test,
        test_fraction=cfg.test_fraction,
        splits_dir=cfg.splits_dir,
        seed=cfg.seed,
    )

    for label in labels:
        label_dir = os.path.join(cfg.data_dir, label)
        all_csvs = [
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.endswith(".csv")
        ]

        test_csvs = set(test_split.get(label, []))
        train_csvs = [p for p in all_csvs if p not in test_csvs]

        if not train_csvs:
            raise RuntimeError(f"No training CSVs left for class '{label}'")

        train_single_class(
            label=label,
            train_csvs=train_csvs,
            test_csvs=test_csvs,
            cfg=cfg,
        )




if __name__ == "__main__":
    cfg = Stage1Config(
        data_dir="data",
        feature_cols=[
            "10_H0", "10_H1", "10_H2", "10_H3", "10_K2", "10_K3",
            "1_A0", "1_A1", "1_A2", "1_A3", "1_K0", "1_K1",
            "1_P0", "1_P1", "1_P2", "1_P3",
            "201_B0", "201_B1", "201_B2", "201_B3", "201_L0", "201_L1",
            "202_F0", "202_F1", "202_F2", "202_F3", "202_N0", "202_N1",
            "301_C0", "301_C1", "301_C2", "301_C3", "301_L2", "301_L3",
            "302_G0", "302_G1", "302_G2", "302_G3", "302_N2", "302_N3",
            "401_D0", "401_D1", "401_D2", "401_D3", "401_M0", "401_M1",
            "402_I0", "402_I1", "402_I2", "402_I3", "402_O0", "402_O1",
            "501_E0", "501_E1", "501_E2", "501_E3", "501_M2", "501_M3",
            "502_J0", "502_J1", "502_J2", "502_J3", "502_O2", "502_O3",
            "avg_1", "avg_10", "avg_201", "avg_202",
            "avg_301", "avg_302", "avg_401", "avg_402",
            "avg_501", "avg_502",
        ],
        delta=25,
        epochs=200,
        lr=1e-4,
        batch_size=128,
        models_dir="models",
        n_test=1,
        test_fraction=0.2
    )

    train_stage1(cfg)
