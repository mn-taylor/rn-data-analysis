"""
Channel group ablation analysis for time series classification models.

Ablates (zeroes-out) specified channel groups to quantify each group's
contribution to model performance.  Groups are defined by the numeric prefix
of column names (e.g., "301_something" matches prefix 301).

Three conditions are evaluated per group:
  - Baseline               : all channels present
  - ablate_<group>         : the group's channels are zeroed out
  - keep_<group>           : all OTHER channels are zeroed out (group-only)

Additionally, all groups are ablated together for a combined condition.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Channel group identification
# ---------------------------------------------------------------------------

def identify_channel_groups(
    feature_names: List[str],
    channel_groups_cfg: Dict[str, dict],
) -> Dict[str, List[int]]:
    """Identify channel indices belonging to each group by numeric prefix.

    A channel name matches a prefix when the name begins with that prefix
    followed by a non-digit character (e.g. an underscore or letter).
    This avoids "301" matching "3010_foo".

    Args:
        feature_names: Ordered list of feature/channel names from the CSV.
        channel_groups_cfg: Dict loaded from the YAML ``channel_groups``
            section.  Each value must have a ``prefixes`` list, e.g.::

                prostate:
                  display_name: "Prostate (OR51E1/OR51E2)"
                  prefixes: [301, 302, 401]

    Returns:
        Dict mapping group_name -> sorted list of matched channel indices.
    """
    group_indices: Dict[str, List[int]] = {}

    for group_name, group_cfg in channel_groups_cfg.items():
        prefixes = [str(p) for p in group_cfg.get("prefixes", [])]
        display = group_cfg.get("display_name", group_name)
        indices: List[int] = []

        for i, name in enumerate(feature_names):
            for pfx in prefixes:
                # Match the prefix number anywhere in the name, but only as a
                # complete numeric token (not as part of a longer number).
                # e.g. "301_avg", "Avg_301", "avg_301_resistance" all match
                # prefix 301, but "3010_foo" does not.
                if re.search(rf"(?<!\d){re.escape(pfx)}(?!\d)", name):
                    indices.append(i)
                    break  # each channel belongs to at most one prefix match

        group_indices[group_name] = sorted(indices)
        samples = [feature_names[i] for i in indices[:4]]
        ellipsis = "..." if len(indices) > 4 else ""
        print(f"  {display}: {len(indices)} channels  {samples}{ellipsis}")

    return group_indices


# ---------------------------------------------------------------------------
# Core ablation evaluation
# ---------------------------------------------------------------------------

def evaluate_with_ablation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    ablate_indices: List[int],
    metric: str = "accuracy",
) -> float:
    """Evaluate model with the specified channels zeroed out.

    Args:
        model: Trained PyTorch model (eval mode expected).
        dataloader: Test DataLoader.
        device: Torch device string.
        ablate_indices: Channel indices to set to zero in every sample.
        metric: "accuracy" or "auc".

    Returns:
        Scalar performance score after ablation.
    """
    from .feature_importance import _evaluate_model

    if not ablate_indices:
        return _evaluate_model(model, dataloader, device, metric)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X_abl = X.clone()

            if X.dim() == 3:                    # (B, C, T)
                for idx in ablate_indices:
                    X_abl[:, idx, :] = 0.0
            elif X.dim() == 2:                  # (B, C)
                for idx in ablate_indices:
                    X_abl[:, idx] = 0.0

            logits = model(X_abl)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

    import numpy as _np
    preds = _np.concatenate(all_preds)
    labels = _np.concatenate(all_labels)
    probs = _np.concatenate(all_probs)

    if metric == "accuracy":
        return (preds == labels).mean()
    elif metric == "auc":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, probs)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Full ablation experiment for one model
# ---------------------------------------------------------------------------

def run_channel_ablation(
    model: nn.Module,
    dataloader: DataLoader,
    feature_names: List[str],
    group_indices: Dict[str, List[int]],
    channel_groups_cfg: Dict[str, dict],
    device: str,
    metric: str = "accuracy",
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Run ablation experiment for all channel groups on a single model.

    Evaluates:
      * ``baseline``           — all channels present
      * ``ablate_<group>``     — group zeroed, rest intact
      * ``keep_<group>``       — only group kept, rest zeroed
      * ``ablate_all_groups``  — all defined groups zeroed simultaneously

    Args:
        model: Trained model (moved to device internally).
        dataloader: Test DataLoader.
        feature_names: Full list of feature names.
        group_indices: Output of :func:`identify_channel_groups`.
        channel_groups_cfg: Raw YAML ``channel_groups`` dict (for display names).
        device: Torch device string.
        metric: "accuracy" or "auc".

    Returns:
        (results, display_names) where ``results`` maps condition key ->
        scalar score and ``display_names`` maps condition key -> human-readable
        label for plots.
    """
    from .feature_importance import _evaluate_model

    model = model.to(device)
    model.eval()

    results: Dict[str, float] = {}
    display_names: Dict[str, str] = {}

    # -- Baseline --
    results["baseline"] = _evaluate_model(model, dataloader, device, metric)
    display_names["baseline"] = "Baseline\n(all channels)"
    print(f"  baseline {metric}: {results['baseline']:.4f}")

    n_channels = len(feature_names)

    for group_name, indices in group_indices.items():
        disp = channel_groups_cfg[group_name].get("display_name", group_name)

        # Ablate this group (zero it, keep the rest)
        key_abl = f"ablate_{group_name}"
        results[key_abl] = evaluate_with_ablation(
            model, dataloader, device, indices, metric
        )
        display_names[key_abl] = f"Ablate\n{disp}"
        drop = results["baseline"] - results[key_abl]
        print(
            f"  {key_abl} ({len(indices)} ch): "
            f"{results[key_abl]:.4f}  (drop {drop:+.4f})"
        )

        # Keep only this group (zero everything else)
        key_keep = f"keep_{group_name}"
        other_indices = [i for i in range(n_channels) if i not in set(indices)]
        results[key_keep] = evaluate_with_ablation(
            model, dataloader, device, other_indices, metric
        )
        display_names[key_keep] = f"Only\n{disp}"
        print(
            f"  {key_keep} ({len(indices)} ch): {results[key_keep]:.4f}"
        )

    # -- Ablate all groups simultaneously --
    all_group_indices = sorted(
        set(i for idxs in group_indices.values() for i in idxs)
    )
    if all_group_indices:
        key_all = "ablate_all_groups"
        results[key_all] = evaluate_with_ablation(
            model, dataloader, device, all_group_indices, metric
        )
        display_names[key_all] = "Ablate\nall groups"
        drop = results["baseline"] - results[key_all]
        print(
            f"  {key_all} ({len(all_group_indices)} ch): "
            f"{results[key_all]:.4f}  (drop {drop:+.4f})"
        )

    return results, display_names


# ---------------------------------------------------------------------------
# Aggregate across folds
# ---------------------------------------------------------------------------

def aggregate_ablation_across_folds(
    fold_results: List[Tuple[Dict[str, float], Dict[str, str]]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, str]]:
    """Aggregate per-fold ablation scores to (mean, std) per condition.

    Args:
        fold_results: List of (results_dict, display_names_dict) from each fold.

    Returns:
        (aggregated, display_names) where ``aggregated`` maps condition key ->
        (mean_score, std_score).
    """
    all_conditions: set = set()
    for res, _ in fold_results:
        all_conditions.update(res.keys())

    aggregated: Dict[str, Tuple[float, float]] = {}
    for cond in all_conditions:
        scores = [res[cond] for res, _ in fold_results if cond in res]
        aggregated[cond] = (float(np.mean(scores)), float(np.std(scores)))

    display_names = fold_results[0][1] if fold_results else {}
    return aggregated, display_names


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_CONDITION_COLORS = {
    "baseline":         "#2196F3",   # blue
    "ablate_all_groups": "#B71C1C",  # dark red
}
_ABLATE_COLOR  = "#FF5722"  # deep orange
_KEEP_COLOR    = "#4CAF50"  # green


def plot_ablation_results(
    ablation_results: Dict[str, Tuple[float, float]],
    display_names: Dict[str, str],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
):
    """Bar plot comparing performance across ablation conditions.

    Args:
        ablation_results: Dict from :func:`aggregate_ablation_across_folds`
            mapping condition -> (mean, std).
        display_names: Human-readable labels per condition key.
        metric: Metric name for the y-axis label.
        save_path: Path to save the figure (None = show interactively).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    conditions = list(ablation_results.keys())
    means = [ablation_results[c][0] for c in conditions]
    stds  = [ablation_results[c][1] for c in conditions]
    labels = [display_names.get(c, c) for c in conditions]

    colors = []
    for c in conditions:
        if c in _CONDITION_COLORS:
            colors.append(_CONDITION_COLORS[c])
        elif c.startswith("ablate_"):
            colors.append(_ABLATE_COLOR)
        elif c.startswith("keep_"):
            colors.append(_KEEP_COLOR)
        else:
            colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(max(7, len(conditions) * 2.0), 5))
    x = np.arange(len(conditions))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric.upper())
    ax.set_title("Channel Group Ablation Results")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (std if std > 0 else 0) + 0.004,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=8,
        )

    # Baseline reference line
    bl = ablation_results.get("baseline")
    if bl is not None:
        ax.axhline(bl[0], color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)

    ax.grid(axis="y", alpha=0.3)
    legend_elements = [
        Patch(facecolor="#2196F3",  label="Baseline (all channels)"),
        Patch(facecolor=_ABLATE_COLOR, label="Group ablated (zeroed out)"),
        Patch(facecolor=_KEEP_COLOR,   label="Only this group (rest zeroed)"),
        Patch(facecolor="#B71C1C",  label="All groups ablated"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ablation plot to {save_path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_ablation_results(
    ablation_results: Dict[str, Tuple[float, float]],
    display_names: Dict[str, str],
    feature_names: List[str],
    group_indices: Dict[str, List[int]],
    metric: str,
    save_dir: str,
    prefix: str,
):
    """Save ablation scores and channel group membership to CSV.

    Writes:
        {prefix}_ablation_results.csv  — one row per condition
        {prefix}_channel_groups.csv    — one row per channel, with group label
    """
    import pandas as pd

    rows = [
        {
            "condition": cond,
            "display_name": display_names.get(cond, cond),
            f"{metric}_mean": mean,
            f"{metric}_std": std,
        }
        for cond, (mean, std) in ablation_results.items()
    ]
    results_path = os.path.join(save_dir, f"{prefix}_ablation_results.csv")
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved ablation results to {results_path}")

    # Channel group membership
    group_rows = [
        {
            "group": grp,
            "channel_index": idx,
            "channel_name": feature_names[idx],
        }
        for grp, indices in group_indices.items()
        for idx in indices
    ]
    if group_rows:
        groups_path = os.path.join(save_dir, f"{prefix}_channel_groups.csv")
        pd.DataFrame(group_rows).to_csv(groups_path, index=False)
        print(f"Saved channel group membership to {groups_path}")
