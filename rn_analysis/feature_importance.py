"""
Feature importance analysis for time series classification models.

This module provides methods for analyzing which features are most important
for model predictions:
- Permutation importance (model-agnostic)
- Integrated gradients (for neural networks)
- Temporal importance analysis (feature importance over time)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm


def compute_permutation_importance(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    n_repeats: int = 10,
    metric: str = "accuracy",
) -> Dict[str, np.ndarray]:
    """Compute permutation feature importance.

    Shuffles each feature channel independently and measures the drop in
    model performance. Higher drop indicates more important feature.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with test data
        feature_names: List of feature names (length = num_channels)
        device: Device to run on ("cpu" or "cuda")
        n_repeats: Number of times to repeat permutation per feature
        metric: Metric to use ("accuracy" or "auc")

    Returns:
        Dictionary with:
            - "importances": (n_features,) mean importance scores
            - "importances_std": (n_features,) std of importance scores
            - "baseline_score": Baseline model score without permutation
    """
    model.eval()
    model = model.to(device)

    # Get baseline score
    baseline_score = _evaluate_model(model, dataloader, device, metric)

    n_features = len(feature_names)
    importance_scores = np.zeros((n_features, n_repeats))

    print(f"Baseline {metric}: {baseline_score:.4f}")
    print(f"\nComputing permutation importance for {n_features} features...")

    # Track some permuted scores for debugging
    debug_scores = []

    for feat_idx in tqdm(range(n_features), desc="Features"):
        for repeat_idx in range(n_repeats):
            # Permute this feature and evaluate
            score = _evaluate_model_with_permutation(
                model, dataloader, device, feat_idx, metric
            )
            # Importance = drop in performance
            importance_scores[feat_idx, repeat_idx] = baseline_score - score

            # Store first 3 features' scores for debugging
            if feat_idx < 3 and repeat_idx == 0:
                debug_scores.append((feature_names[feat_idx], score, baseline_score - score))

    # Print debug info
    if debug_scores:
        print("\nDebug: First 3 features permutation results:")
        for fname, perm_score, importance in debug_scores:
            print(f"  {fname}: permuted_score={perm_score:.4f}, importance={importance:.4f}")

    mean_importances = importance_scores.mean(axis=1)
    print(f"\nImportance statistics:")
    print(f"  Min: {mean_importances.min():.6f}")
    print(f"  Max: {mean_importances.max():.6f}")
    print(f"  Mean: {mean_importances.mean():.6f}")
    print(f"  Non-zero features: {(mean_importances > 1e-6).sum()} / {n_features}")

    return {
        "importances": mean_importances,
        "importances_std": importance_scores.std(axis=1),
        "baseline_score": baseline_score,
        "feature_names": feature_names,
    }


def _evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    metric: str = "accuracy",
) -> float:
    """Evaluate model on dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    if metric == "accuracy":
        return (preds == labels).mean()
    elif metric == "auc":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, probs)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _evaluate_model_with_permutation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    feature_idx: int,
    metric: str = "accuracy",
) -> float:
    """Evaluate model with one feature permuted.

    Note: Assumes data is in (B, C, T) format where C is the channel/feature dimension.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Permute the feature across samples in batch
            # Data should be in (B, C, T) format from the dataloader
            X_perm = X.clone()
            if X.dim() == 3:
                # For (B, C, T) format, permute along channel dimension (dim=1)
                perm_idx = torch.randperm(X.shape[0], device=device)
                X_perm[:, feature_idx, :] = X[perm_idx, feature_idx, :]
            elif X.dim() == 2:
                # For (B, C) format
                perm_idx = torch.randperm(X.shape[0], device=device)
                X_perm[:, feature_idx] = X[perm_idx, feature_idx]
            else:
                raise ValueError(f"Unexpected input dimension: {X.dim()}")

            logits = model(X_perm)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    if metric == "accuracy":
        return (preds == labels).mean()
    elif metric == "auc":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, probs)


def compute_integrated_gradients(
    model: nn.Module,
    X: torch.Tensor,
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    device: str = "cpu",
) -> np.ndarray:
    """Compute Integrated Gradients for feature attribution.

    Integrated Gradients computes the gradient of the output with respect to
    each input feature, integrated along a path from a baseline to the input.

    Args:
        model: Trained PyTorch model
        X: Input tensor of shape (B, C, T) or (B, T, C)
        target_class: Class to compute gradients for
        baseline: Baseline input (default: zeros)
        n_steps: Number of integration steps
        device: Device to run on

    Returns:
        Attributions array of shape (B, C, T) or (B, T, C)
    """
    model.eval()
    model = model.to(device)
    X = X.to(device)

    if baseline is None:
        baseline = torch.zeros_like(X)
    else:
        baseline = baseline.to(device)

    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)

    # Storage for gradients
    integrated_grads = torch.zeros_like(X)

    for alpha in alphas:
        # Interpolate between baseline and input
        X_interp = baseline + alpha * (X - baseline)
        X_interp.requires_grad_(True)

        # Forward pass
        output = model(X_interp)

        # Compute gradient w.r.t. target class
        model.zero_grad()
        target_score = output[:, target_class].sum()
        target_score.backward()

        # Accumulate gradients
        integrated_grads += X_interp.grad.detach()

    # Average gradients and multiply by input difference
    integrated_grads = integrated_grads / (n_steps + 1)
    attributions = (X - baseline) * integrated_grads

    return attributions.cpu().numpy()


def compute_temporal_importance(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    method: str = "gradient",
    target_class: int = 1,
) -> Dict[str, np.ndarray]:
    """Compute importance of each feature at each time step.

    This shows which features are important at which points in time.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with test data
        feature_names: List of feature names
        device: Device to run on
        method: Method to use ("gradient" or "integrated_gradients")
        target_class: Target class for gradient computation

    Returns:
        Dictionary with:
            - "temporal_importance": (C, T) array of average importance
            - "feature_names": List of feature names
    """
    model.eval()
    model = model.to(device)

    all_attributions = []

    print(f"Computing temporal importance using {method}...")

    for X, y in tqdm(dataloader, desc="Batches"):
        X = X.to(device)

        if method == "integrated_gradients":
            attr = compute_integrated_gradients(
                model, X, target_class, device=device
            )
        elif method == "gradient":
            attr = _compute_gradient_attribution(
                model, X, target_class, device
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Average over batch and take absolute value
        # attr shape: (B, C, T) or (B, T, C)
        if attr.shape[1] > attr.shape[2]:  # (B, C, T)
            attr_avg = np.abs(attr).mean(axis=0)  # (C, T)
        else:  # (B, T, C)
            attr_avg = np.abs(attr).mean(axis=0).T  # (C, T)

        all_attributions.append(attr_avg)

    # Average across all batches
    temporal_importance = np.mean(all_attributions, axis=0)

    return {
        "temporal_importance": temporal_importance,
        "feature_names": feature_names,
    }


def _compute_gradient_attribution(
    model: nn.Module,
    X: torch.Tensor,
    target_class: int,
    device: str,
) -> np.ndarray:
    """Compute simple gradient-based attribution."""
    X.requires_grad_(True)

    output = model(X)
    model.zero_grad()

    target_score = output[:, target_class].sum()
    target_score.backward()

    grads = X.grad.detach()
    attributions = (X * grads).cpu().numpy()

    return attributions


def aggregate_importance_across_folds(
    fold_results: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Aggregate feature importance across k-folds.

    Args:
        fold_results: List of importance dictionaries from each fold

    Returns:
        Dictionary with aggregated results:
            - "importances_mean": Mean importance across folds
            - "importances_std": Std of importance across folds
            - "feature_names": List of feature names
    """
    n_folds = len(fold_results)
    n_features = len(fold_results[0]["importances"])

    all_importances = np.zeros((n_folds, n_features))

    for i, result in enumerate(fold_results):
        all_importances[i] = result["importances"]

    return {
        "importances_mean": all_importances.mean(axis=0),
        "importances_std": all_importances.std(axis=0),
        "feature_names": fold_results[0]["feature_names"],
    }


def plot_feature_importance(
    importance_dict: Dict[str, np.ndarray],
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """Plot feature importance bar chart.

    Args:
        importance_dict: Dictionary with "importances"/"importances_mean", "importances_std", "feature_names"
        top_k: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure (None = display only)
    """
    import matplotlib.pyplot as plt

    # Handle both single-fold results (with "importances") and aggregated results (with "importances_mean")
    if "importances_mean" in importance_dict:
        importances = importance_dict["importances_mean"]
    elif "importances" in importance_dict:
        importances = importance_dict["importances"]
    else:
        raise KeyError("importance_dict must contain either 'importances' or 'importances_mean'")

    feature_names = importance_dict["feature_names"]

    # Get std if available
    if "importances_std" in importance_dict:
        stds = importance_dict["importances_std"]
    else:
        stds = None

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_idx))

    if stds is not None:
        ax.barh(y_pos, importances[sorted_idx], xerr=stds[sorted_idx], capsize=3)
    else:
        ax.barh(y_pos, importances[sorted_idx])

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()


def plot_temporal_importance(
    temporal_dict: Dict[str, np.ndarray],
    top_k: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot temporal importance heatmap.

    Shows importance of each feature over time.

    Args:
        temporal_dict: Dictionary with "temporal_importance" and "feature_names"
        top_k: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    temporal_importance = temporal_dict["temporal_importance"]  # (C, T)
    feature_names = temporal_dict["feature_names"]

    # Get top k features by total importance
    total_importance = temporal_importance.sum(axis=1)
    top_idx = np.argsort(total_importance)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        temporal_importance[top_idx],
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Features")
    ax.set_title(f"Temporal Feature Importance (Top {top_k} Features)")

    plt.colorbar(im, ax=ax, label="Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved temporal importance plot to {save_path}")
    else:
        plt.show()


def save_importance_results(
    importance_dict: Dict[str, np.ndarray],
    save_path: str,
):
    """Save feature importance results to CSV.

    Args:
        importance_dict: Dictionary with importance results
        save_path: Path to save CSV file
    """
    import pandas as pd

    # Handle both single-fold results (with "importances") and aggregated results (with "importances_mean")
    if "importances_mean" in importance_dict:
        df = pd.DataFrame({
            "feature": importance_dict["feature_names"],
            "importance_mean": importance_dict["importances_mean"],
        })
        if "importances_std" in importance_dict:
            df["importance_std"] = importance_dict["importances_std"]
    elif "importances" in importance_dict:
        df = pd.DataFrame({
            "feature": importance_dict["feature_names"],
            "importance": importance_dict["importances"],
        })
        if "importances_std" in importance_dict:
            df["importance_std"] = importance_dict["importances_std"]
    else:
        raise KeyError("importance_dict must contain either 'importances' or 'importances_mean'")

    df = df.sort_values(by=df.columns[1], ascending=False)  # Sort by importance column (1st data column)
    df.to_csv(save_path, index=False)
    print(f"Saved feature importance results to {save_path}")
