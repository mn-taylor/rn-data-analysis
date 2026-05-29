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


def _build_permuted_mega_batch(
    X: torch.Tensor, feat_indices: List[int], n_repeats: int, device: str
) -> torch.Tensor:
    """Stack n_chunk * n_repeats permuted copies of X into one mega-batch.

    Layout: (feat0,rep0), (feat0,rep1), ..., (feat1,rep0), ... each of size B.
    Returns shape: (n_chunk * n_repeats * B, C, T).
    """
    copies = []
    for feat_idx in feat_indices:
        for _ in range(n_repeats):
            X_p = X.clone()
            perm = torch.randperm(X.shape[0], device=device)
            if X.dim() == 3:
                X_p[:, feat_idx, :] = X[perm, feat_idx, :]
            else:
                X_p[:, feat_idx] = X[perm, feat_idx]
            copies.append(X_p)
    return torch.cat(copies, dim=0)


def compute_permutation_importance(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    n_repeats: int = 10,
    metric: str = "accuracy",
    n_parallel_features: int = 1,
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
        n_parallel_features: Number of features to permute simultaneously.
            >1 stacks n_parallel_features * n_repeats permuted copies of each
            batch into a single forward pass, reducing total GPU launches by
            ~n_parallel_features×. Tune to GPU memory:
              4  → ~360 MB extra per batch (B=32)
              8  → ~720 MB
              16 → ~1.4 GB

    Returns:
        Dictionary with:
            - "importances": (n_features,) mean importance scores
            - "importances_std": (n_features,) std of importance scores
            - "baseline_score": Baseline model score without permutation
    """
    model.eval()
    model = model.to(device)

    baseline_score = _evaluate_model(model, dataloader, device, metric)

    n_features = len(feature_names)
    importance_scores = np.zeros((n_features, n_repeats))

    print(f"Baseline {metric}: {baseline_score:.4f}")
    print(f"\nComputing permutation importance for {n_features} features "
          f"(n_parallel_features={n_parallel_features})...")

    if n_parallel_features <= 1:
        # --- sequential path (original) ---
        debug_scores = []
        for feat_idx in tqdm(range(n_features), desc="Features"):
            for repeat_idx in range(n_repeats):
                score = _evaluate_model_with_permutation(
                    model, dataloader, device, feat_idx, metric
                )
                importance_scores[feat_idx, repeat_idx] = baseline_score - score
                if feat_idx < 3 and repeat_idx == 0:
                    debug_scores.append((feature_names[feat_idx], score, baseline_score - score))

        if debug_scores:
            print("\nDebug: First 3 features permutation results:")
            for fname, perm_score, importance in debug_scores:
                print(f"  {fname}: permuted_score={perm_score:.4f}, importance={importance:.4f}")

    else:
        # --- vectorized path: process n_parallel_features at once ---
        from sklearn.metrics import roc_auc_score as _roc_auc

        n_chunks = (n_features + n_parallel_features - 1) // n_parallel_features
        for chunk_start in tqdm(range(0, n_features, n_parallel_features), desc="Feature chunks",
                                total=n_chunks):
            chunk_feats = list(range(chunk_start,
                                     min(chunk_start + n_parallel_features, n_features)))
            n_chunk = len(chunk_feats)

            # slot_preds/probs[i][j] → list of arrays from each batch
            slot_preds = [[[] for _ in range(n_repeats)] for _ in range(n_chunk)]
            slot_probs = [[[] for _ in range(n_repeats)] for _ in range(n_chunk)]
            all_labels = []

            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                B = X.shape[0]

                mega = _build_permuted_mega_batch(X, chunk_feats, n_repeats, device)
                with torch.no_grad():
                    logits = model(mega)  # (n_chunk * n_repeats * B, n_classes)
                probs_mega = torch.softmax(logits, dim=1)
                preds_mega = logits.argmax(dim=1)

                all_labels.append(y.cpu().numpy())

                for i in range(n_chunk):
                    for j in range(n_repeats):
                        slot = i * n_repeats + j
                        s, e = slot * B, slot * B + B
                        slot_preds[i][j].append(preds_mega[s:e].cpu().numpy())
                        slot_probs[i][j].append(probs_mega[s:e, 1].cpu().numpy())

            labels = np.concatenate(all_labels)
            for i, feat_idx in enumerate(chunk_feats):
                for j in range(n_repeats):
                    preds_ij = np.concatenate(slot_preds[i][j])
                    probs_ij = np.concatenate(slot_probs[i][j])
                    if metric == "accuracy":
                        score = (preds_ij == labels).mean()
                    elif metric == "auc":
                        score = _roc_auc(labels, probs_ij)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    importance_scores[feat_idx, j] = baseline_score - score

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


# =============================================================================
# Occlusion Importance
# =============================================================================

def _build_occluded_mega_batch(
    X: torch.Tensor, feat_indices: List[int], channel_means: np.ndarray
) -> torch.Tensor:
    """Stack n_chunk occluded copies of X into one mega-batch.

    Layout: feat0, feat1, ..., each of size B. Returns (n_chunk * B, C, T).
    """
    copies = []
    for feat_idx in feat_indices:
        X_occ = X.clone()
        if X.dim() == 3:
            X_occ[:, feat_idx, :] = float(channel_means[feat_idx])
        else:
            X_occ[:, feat_idx] = float(channel_means[feat_idx])
        copies.append(X_occ)
    return torch.cat(copies, dim=0)


def compute_occlusion_importance(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    metric: str = "accuracy",
    n_parallel_features: int = 1,
) -> Dict[str, np.ndarray]:
    """Occlusion sensitivity: mask each feature channel with its dataset mean.

    More deterministic than permutation importance — replaces the channel with
    a constant (its mean across the test set) rather than shuffling across
    samples.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with test data
        feature_names: List of feature names (length = num_channels)
        device: Device to run on
        metric: Metric to use ("accuracy" or "auc")
        n_parallel_features: Number of features to occlude simultaneously.
            >1 stacks n_parallel_features occluded copies of each batch into a
            single forward pass (~N× speedup). Less memory pressure than
            permutation (no repeats dimension): 4→~72 MB, 8→~144 MB, 16→~288 MB
            extra per batch (B=32).

    Returns:
        Dictionary with "importances", "importances_std", "baseline_score",
        "feature_names".
    """
    model.eval()
    model = model.to(device)

    channel_means = _compute_channel_means(dataloader, device)
    baseline_score = _evaluate_model(model, dataloader, device, metric)
    print(f"Baseline {metric}: {baseline_score:.4f}")
    print(f"\nComputing occlusion importance for {len(feature_names)} features "
          f"(n_parallel_features={n_parallel_features})...")

    n_features = len(feature_names)
    importance_scores = np.zeros(n_features)

    if n_parallel_features <= 1:
        # --- sequential path (original) ---
        for feat_idx in tqdm(range(n_features), desc="Occlusion"):
            fill = float(channel_means[feat_idx])
            score = _evaluate_model_with_occlusion(
                model, dataloader, device, feat_idx, metric, fill_value=fill
            )
            importance_scores[feat_idx] = baseline_score - score

    else:
        # --- vectorized path: process n_parallel_features at once ---
        from sklearn.metrics import roc_auc_score as _roc_auc

        n_chunks = (n_features + n_parallel_features - 1) // n_parallel_features
        for chunk_start in tqdm(range(0, n_features, n_parallel_features), desc="Occlusion chunks",
                                total=n_chunks):
            chunk_feats = list(range(chunk_start,
                                     min(chunk_start + n_parallel_features, n_features)))
            n_chunk = len(chunk_feats)

            slot_preds = [[] for _ in range(n_chunk)]
            slot_probs = [[] for _ in range(n_chunk)]
            all_labels = []

            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                B = X.shape[0]

                mega = _build_occluded_mega_batch(X, chunk_feats, channel_means)
                with torch.no_grad():
                    logits = model(mega)  # (n_chunk * B, n_classes)
                probs_mega = torch.softmax(logits, dim=1)
                preds_mega = logits.argmax(dim=1)

                all_labels.append(y.cpu().numpy())

                for i in range(n_chunk):
                    s, e = i * B, i * B + B
                    slot_preds[i].append(preds_mega[s:e].cpu().numpy())
                    slot_probs[i].append(probs_mega[s:e, 1].cpu().numpy())

            labels = np.concatenate(all_labels)
            for i, feat_idx in enumerate(chunk_feats):
                preds_i = np.concatenate(slot_preds[i])
                probs_i = np.concatenate(slot_probs[i])
                if metric == "accuracy":
                    score = (preds_i == labels).mean()
                elif metric == "auc":
                    score = _roc_auc(labels, probs_i)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                importance_scores[feat_idx] = baseline_score - score

    return {
        "importances": importance_scores,
        "importances_std": np.zeros(n_features),
        "baseline_score": baseline_score,
        "feature_names": feature_names,
    }


def _compute_channel_means(dataloader: DataLoader, device: str) -> np.ndarray:
    """Compute per-channel mean across all data (for occlusion fill values)."""
    all_data = []
    with torch.no_grad():
        for X, _ in dataloader:
            all_data.append(X.cpu())
    all_data = torch.cat(all_data, dim=0)
    if all_data.dim() == 3:   # (N, C, T)
        return all_data.mean(dim=(0, 2)).numpy()   # (C,)
    return all_data.mean(dim=0).numpy()            # (N, C) -> (C,)


def _evaluate_model_with_occlusion(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    feature_idx: int,
    metric: str = "accuracy",
    fill_value: float = 0.0,
) -> float:
    """Evaluate model with one channel replaced by a constant fill value."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X_occ = X.clone()
            if X.dim() == 3:
                X_occ[:, feature_idx, :] = fill_value
            else:
                X_occ[:, feature_idx] = fill_value

            logits = model(X_occ)
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


# =============================================================================
# Integrated Gradients — channel-level aggregation
# =============================================================================

def compute_ig_channel_importance(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    n_steps: int = 50,
    target_class: int = 1,
    n_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Channel-level importance via Integrated Gradients.

    Runs IG per batch (giving (B, C, T) attributions), takes the absolute
    value, and averages over time and batch to produce a (C,) importance
    vector comparable with permutation / occlusion scores.

    Args:
        model: Trained PyTorch model (must support gradient flow)
        dataloader: DataLoader with test data
        feature_names: List of feature names
        device: Device to run on
        n_steps: Integration steps along the IG path
        target_class: Class index to attribute towards
        n_samples: Maximum number of samples to process (None = all)

    Returns:
        Dictionary with "importances", "importances_std", "baseline_score",
        "feature_names".
    """
    model.eval()
    model = model.to(device)

    batch_importances: List[np.ndarray] = []
    sample_count = 0

    for X, _ in tqdm(dataloader, desc="IG"):
        if n_samples is not None and sample_count >= n_samples:
            break
        X = X.to(device)
        attr = compute_integrated_gradients(
            model, X, target_class, n_steps=n_steps, device=device
        )
        # attr: (B, C, T) -> aggregate over batch and time
        if attr.ndim == 3:
            channel_attr = np.abs(attr).mean(axis=(0, 2))   # (C,)
        else:
            channel_attr = np.abs(attr).mean(axis=0)

        batch_importances.append(channel_attr)
        sample_count += X.shape[0]

    mean_imp = np.mean(batch_importances, axis=0)
    std_imp = np.std(batch_importances, axis=0)

    return {
        "importances": mean_imp,
        "importances_std": std_imp,
        "baseline_score": 0.0,
        "feature_names": feature_names,
    }


# =============================================================================
# SHAP Importance
# =============================================================================

def compute_shap_importance(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    n_background: int = 50,
    n_samples: int = 100,
    target_class: int = 1,
) -> Dict[str, np.ndarray]:
    """Channel-level importance via SHAP GradientExplainer.

    Requires the shap package (``pip install shap``).  The GradientExplainer
    integrates gradients with respect to a background distribution, similar
    to Integrated Gradients but averaged over many reference points.

    Note: Not compatible with non-differentiable models (e.g. ROCKET).

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader with test data
        feature_names: List of feature names
        device: Device to run on
        n_background: Number of background reference samples
        n_samples: Number of test samples to explain
        target_class: Class index to attribute towards

    Returns:
        Dictionary with "importances", "importances_std", "baseline_score",
        "feature_names".
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Run: pip install shap")

    model.eval()

    # Collect enough data for background + test
    all_batches: List[torch.Tensor] = []
    count = 0
    for X, _ in dataloader:
        all_batches.append(X)
        count += X.shape[0]
        if count >= n_background + n_samples:
            break

    all_data = torch.cat(all_batches, dim=0)
    background = all_data[:n_background].to(device)
    test_data = all_data[n_background: n_background + n_samples]
    if test_data.shape[0] == 0:
        test_data = background
    test_data = test_data.to(device)

    print(f"  SHAP: {background.shape[0]} background, {test_data.shape[0]} test samples")

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_data)

    # Handle output format differences across shap versions
    if isinstance(shap_values, list):
        attrs = np.abs(shap_values[target_class])
    elif hasattr(shap_values, "values"):
        sv = shap_values.values
        if sv.ndim == 4:           # (n_samples, C, T, n_classes)
            attrs = np.abs(sv[..., target_class])
        else:
            attrs = np.abs(sv)
    else:
        attrs = np.abs(np.array(shap_values))

    if attrs.ndim == 3:            # (n_samples, C, T)
        channel_importance = attrs.mean(axis=(0, 2))
        channel_std = attrs.std(axis=0).mean(axis=1)
    elif attrs.ndim == 2:          # (n_samples, C)
        channel_importance = attrs.mean(axis=0)
        channel_std = attrs.std(axis=0)
    else:
        raise ValueError(f"Unexpected SHAP attribution shape: {attrs.shape}")

    return {
        "importances": channel_importance,
        "importances_std": channel_std,
        "baseline_score": 0.0,
        "feature_names": feature_names,
    }


# =============================================================================
# Multi-method comparison
# =============================================================================

def compare_importance_methods(
    method_results: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute Spearman ρ matrix between importance methods.

    Args:
        method_results: Dict of {method_name: importance_dict}

    Returns:
        (corr_matrix, pval_matrix, method_names)
        corr_matrix and pval_matrix are (n_methods, n_methods) arrays.
    """
    from scipy.stats import spearmanr

    method_names = list(method_results.keys())
    n = len(method_names)

    vecs: Dict[str, np.ndarray] = {}
    for name, result in method_results.items():
        vecs[name] = result.get("importances_mean", result.get("importances"))

    corr = np.eye(n)
    pval = np.zeros((n, n))

    for i, ni in enumerate(method_names):
        for j, nj in enumerate(method_names):
            if i != j:
                rho, p = spearmanr(vecs[ni], vecs[nj])
                corr[i, j] = rho
                pval[i, j] = p

    return corr, pval, method_names


def plot_method_comparison(
    corr_matrix: np.ndarray,
    pval_matrix: np.ndarray,
    method_names: List[str],
    save_path: Optional[str] = None,
):
    """Heatmap of Spearman ρ between importance methods.

    Cells with p < 0.05 (off-diagonal) are annotated with an asterisk.
    """
    import matplotlib.pyplot as plt

    n = len(method_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.5), max(4, n * 1.3)))

    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.set_yticklabels(method_names)
    ax.set_title("Spearman ρ Between Importance Methods")

    for i in range(n):
        for j in range(n):
            sig = "*" if i != j and pval_matrix[i, j] < 0.05 else ""
            txt = f"{corr_matrix[i, j]:.2f}{sig}"
            color = "white" if abs(corr_matrix[i, j]) > 0.7 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved method correlation heatmap to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_multi_method_importance(
    method_results: Dict[str, Dict[str, np.ndarray]],
    feature_names: List[str],
    top_k: int = 20,
    save_path: Optional[str] = None,
):
    """Side-by-side bar charts of feature importance for each method.

    Features are ranked by their average normalised importance across all
    methods so that the same set of top-k features appears in every panel.
    Importance scores within each method are min-max normalised to [0, 1] to
    make panels visually comparable.
    """
    import matplotlib.pyplot as plt

    method_names = list(method_results.keys())
    n_methods = len(method_names)

    # Extract importance vector for each method
    all_imps: List[np.ndarray] = []
    for name in method_names:
        res = method_results[name]
        all_imps.append(res.get("importances_mean", res.get("importances")))

    # Min-max normalise each method
    normalized: List[np.ndarray] = []
    for imp in all_imps:
        lo, hi = imp.min(), imp.max()
        normalized.append((imp - lo) / (hi - lo) if hi > lo else np.zeros_like(imp))

    mean_norm = np.mean(normalized, axis=0)
    top_idx = np.argsort(mean_norm)[::-1][:top_k]
    top_features = [feature_names[i] for i in top_idx]

    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
               "#FF9800", "#00BCD4", "#E91E63", "#607D8B"]

    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(4 * n_methods, max(6, top_k * 0.35)),
        sharey=True,
    )
    if n_methods == 1:
        axes = [axes]

    for ax, name, norm_imp, color in zip(axes, method_names, normalized,
                                          palette[:n_methods]):
        values = norm_imp[top_idx]
        y_pos = np.arange(len(top_idx))
        ax.barh(y_pos, values, color=color, alpha=0.85, edgecolor="none")
        ax.set_yticks(y_pos)
        if ax is axes[0]:
            ax.set_yticklabels(top_features, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Norm. Importance")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Top {top_k} Features — All Methods", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved multi-method importance plot to {save_path}")
        plt.close()
    else:
        plt.show()


def save_method_comparison(
    method_results: Dict[str, Dict[str, np.ndarray]],
    corr_matrix: np.ndarray,
    pval_matrix: np.ndarray,
    method_names: List[str],
    feature_names: List[str],
    save_dir: str,
    prefix: str,
):
    """Save per-feature importances (all methods) and Spearman matrices to CSV.

    Writes three files:
        {prefix}_all_methods.csv       — per-feature importance per method
        {prefix}_method_correlations.csv — Spearman ρ matrix
        {prefix}_method_pvalues.csv    — p-value matrix
    """
    import os
    import pandas as pd

    df = pd.DataFrame({"feature": feature_names})
    for name, res in method_results.items():
        imp = res.get("importances_mean", res.get("importances"))
        df[f"{name}_importance"] = imp
    path = os.path.join(save_dir, f"{prefix}_all_methods.csv")
    df.to_csv(path, index=False)
    print(f"Saved all-methods importances to {path}")

    corr_df = pd.DataFrame(corr_matrix, index=method_names, columns=method_names)
    corr_path = os.path.join(save_dir, f"{prefix}_method_correlations.csv")
    corr_df.to_csv(corr_path)
    print(f"Saved Spearman ρ matrix to {corr_path}")

    pval_df = pd.DataFrame(pval_matrix, index=method_names, columns=method_names)
    pval_path = os.path.join(save_dir, f"{prefix}_method_pvalues.csv")
    pval_df.to_csv(pval_path)
    print(f"Saved p-value matrix to {pval_path}")
