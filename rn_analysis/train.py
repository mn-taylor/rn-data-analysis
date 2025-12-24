"""
Training, validation, and evaluation functions.

This module provides functions for training models, evaluating performance,
and running k-fold cross-validation.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .dataloader import CycleDataset, get_file_labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        grad_clip: Gradient clipping value (None = no clipping)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate for one epoch.

    Args:
        model: Neural network model
        dataloader: Validation/test data loader
        loss_fn: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def compute_auc(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    positive_class: int = 1,
) -> float:
    """Compute ROC AUC score.

    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to evaluate on
        positive_class: Index of positive class

    Returns:
        AUC score
    """
    model.eval()
    y_true = []
    y_score = []

    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        y_true.append(y.cpu().numpy())
        y_score.append(probs[:, positive_class].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    return roc_auc_score(y_true, y_score)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 50,
    grad_clip: Optional[float] = 1.0,
    early_stopping_patience: Optional[int] = 10,
    checkpoint_path: Optional[str] = "best_model.pt",
    print_every: int = 1,
    use_tqdm: bool = True,
) -> Dict[str, List[float]]:
    """Train a model with validation and early stopping.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        epochs: Maximum number of epochs
        grad_clip: Gradient clipping value
        early_stopping_patience: Patience for early stopping (None = no early stopping)
        checkpoint_path: Path to save best model (None = don't save)
        print_every: Print metrics every N epochs (0 = only at end)
        use_tqdm: Use tqdm progress bar

    Returns:
        Dictionary with training history (train_loss, train_acc, val_loss, val_acc, val_auc)
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    best_val_auc = 0
    best_val_acc = 0
    patience_counter = 0

    epoch_iter = tqdm(range(1, epochs + 1), desc="Training") if use_tqdm else range(1, epochs + 1)

    for epoch in epoch_iter:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip
        )

        # Validate
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        val_auc = compute_auc(model, val_loader, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Print progress
        if print_every > 0 and (epoch % print_every == 0 or epoch == 1):
            msg = (
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.4f}"
            )
            if use_tqdm:
                tqdm.write(msg)
            else:
                print(msg)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
                if print_every > 0:
                    msg = f"  → New best val AUC: {best_val_auc:.4f}"
                    if use_tqdm:
                        tqdm.write(msg)
                    else:
                        print(msg)
        else:
            if early_stopping_patience is not None:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    msg = f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)"
                    if use_tqdm:
                        tqdm.write(msg)
                    else:
                        print(msg)
                    break

    # Print final results
    msg = f"\nBest val AUC: {best_val_auc:.4f}, Best val Acc: {best_val_acc:.4f}"
    if use_tqdm:
        tqdm.write(msg)
    else:
        print(msg)

    # Load best model if saved
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        msg = "Loaded best model weights"
        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

    return history


def stratified_kfold_cv(
    model_class: type,
    model_kwargs: Dict[str, Any],
    files: List[str],
    label_map: Dict[str, int],
    n_folds: int = 5,
    T: int = 512,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: str = "cpu",
    seed: int = 42,
    early_stopping_patience: int = 10,
    output_format: str = "channels_first",
) -> Dict[str, Any]:
    """Perform stratified k-fold cross-validation.

    Args:
        model_class: The model class to instantiate
        model_kwargs: Dict of kwargs to pass to model_class
        files: List of file paths
        label_map: Dict mapping folder names to class indices
        n_folds: Number of folds
        T: Time series length
        batch_size: Batch size for training
        epochs: Max epochs per fold
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on
        seed: Random seed
        early_stopping_patience: Patience for early stopping
        output_format: "channels_first" or "time_first"

    Returns:
        Dict with fold results and aggregated metrics
    """
    # Set seeds for reproducibility
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get labels for stratification
    y = get_file_labels(files, label_map)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Track results
    fold_results = []
    all_aucs = []
    all_best_accs = []

    print(f"Starting {n_folds}-fold stratified cross-validation...")
    print(f"Total samples: {len(files)}, Positive: {y.sum()}, Negative: {len(y) - y.sum()}")
    print("=" * 80)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(files, y)):
        print(f"\nFOLD {fold_idx + 1}/{n_folds}")
        print("-" * 80)

        # Split files
        train_files = [files[i] for i in train_idx]
        val_files = [files[i] for i in val_idx]

        # Print fold statistics
        train_labels = y[train_idx]
        val_labels = y[val_idx]
        print(
            f"Train: {len(train_files)} samples "
            f"(Pos: {train_labels.sum()}, Neg: {len(train_labels) - train_labels.sum()})"
        )
        print(
            f"Val:   {len(val_files)} samples "
            f"(Pos: {val_labels.sum()}, Neg: {len(val_labels) - val_labels.sum()})"
        )

        # Create datasets
        train_ds = CycleDataset(train_files, label_map, T=T, output_format=output_format)
        val_ds = CycleDataset(
            val_files, label_map, T=T, feature_cols=train_ds.feature_cols, output_format=output_format
        )

        # Create dataloaders
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Initialize model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_auc = 0
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = train_epoch(model, train_dl, optimizer, loss_fn, device)

            # Validate
            val_loss, val_acc = eval_epoch(model, val_dl, loss_fn, device)
            val_auc = compute_auc(model, val_dl, device)

            # Early stopping based on AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"  Best Val AUC: {best_val_auc:.4f}, Best Val Acc: {best_val_acc:.4f}")

        # Store results
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "best_val_auc": best_val_auc,
                "best_val_acc": best_val_acc,
                "train_size": len(train_files),
                "val_size": len(val_files),
            }
        )
        all_aucs.append(best_val_auc)
        all_best_accs.append(best_val_acc)

    # Aggregate results
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    for res in fold_results:
        print(
            f"Fold {res['fold']}: Val AUC = {res['best_val_auc']:.4f}, "
            f"Val Acc = {res['best_val_acc']:.4f}"
        )

    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_acc = np.mean(all_best_accs)
    std_acc = np.std(all_best_accs)

    print("-" * 80)
    print(f"Mean Val AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean Val Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print("=" * 80)

    return {
        "fold_results": fold_results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "all_aucs": all_aucs,
        "all_accs": all_best_accs,
    }
