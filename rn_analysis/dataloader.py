"""
Dataset classes and data loading utilities.

This module provides PyTorch Dataset classes for time series data and
utility functions for loading and preprocessing CSV files.
"""

import os
import json
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CycleDataset(Dataset):
    """Dataset for cycle-level time series data.

    Each sample is a CSV file containing a single (run_id, cycle) with
    time series sensor data. The data is resampled to a fixed length T
    using linear interpolation.

    Args:
        files: List of CSV file paths
        label_map: Dict mapping folder names to class indices
        T: Fixed time series length (resampled)
        feature_cols: List of feature column names (None = auto-detect)
        output_format: "channels_first" for (C, T) or "time_first" for (T, C)

    Returns:
        X: Tensor of shape (C, T) or (T, C) depending on output_format
        y: Class label (int)
    """

    def __init__(
        self,
        files: List[str],
        label_map: Dict[str, int],
        T: int = 512,
        feature_cols: Optional[List[str]] = None,
        output_format: str = "channels_first",
    ):
        self.files = files
        self.label_map = label_map
        self.T = T
        self.output_format = output_format

        # Infer feature columns from the first file unless provided
        if feature_cols is None:
            df0 = pd.read_csv(files[0])
            id_cols = {"run_id", "cycle", "relative_time_sec", "section", "patient_id"}
            self.feature_cols = [c for c in df0.columns if c not in id_cols]
        else:
            self.feature_cols = list(feature_cols)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        df = pd.read_csv(path).sort_values("relative_time_sec")

        # Extract time and features
        t = df["relative_time_sec"].to_numpy()
        X = df[self.feature_cols].to_numpy(dtype=np.float32)  # (N, C)

        # Resample to fixed length T using linear interpolation
        t_grid = np.linspace(t.min(), t.max(), self.T)

        if self.output_format == "channels_first":
            # (C, T) format - stack channels
            Xr = np.stack(
                [np.interp(t_grid, t, X[:, c]) for c in range(X.shape[1])], axis=0
            )
        else:
            # (T, C) format - stack time steps
            Xr = np.stack(
                [np.interp(t_grid, t, X[:, c]) for c in range(X.shape[1])], axis=1
            )

        # Label from parent folder name
        folder = os.path.basename(os.path.dirname(path))
        y = self.label_map[folder]

        return torch.from_numpy(Xr).float(), torch.tensor(y, dtype=torch.long)


def list_csvs_by_class(
    root: str, class_names: Tuple[str, ...] = ("POSITIVE", "CONTROL")
) -> List[str]:
    """List all CSV files in class subdirectories.

    Args:
        root: Root directory containing class subdirectories
        class_names: Tuple of class folder names

    Returns:
        List of absolute paths to CSV files
    """
    files = []
    for cls in class_names:
        d = os.path.join(root, cls)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
    return files


def get_file_labels(files: List[str], label_map: Dict[str, int]) -> np.ndarray:
    """Extract labels from file paths based on parent folder name.

    Args:
        files: List of file paths
        label_map: Dict mapping folder names to class indices

    Returns:
        Array of class labels
    """
    labels = []
    for f in files:
        folder = os.path.basename(os.path.dirname(f))
        labels.append(label_map[folder])
    return np.array(labels)


def train_test_split_files(
    files: List[str],
    train_split: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Split files into train and test sets.

    Args:
        files: List of file paths
        train_split: Fraction of data for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, test_files)
    """
    if seed is not None:
        rng = random.Random(seed)
        files = files.copy()
        rng.shuffle(files)
    else:
        files = files.copy()
        random.shuffle(files)

    split = int(train_split * len(files))
    return files[:split], files[split:]


def create_dataloaders(
    train_files: List[str],
    test_files: List[str],
    label_map: Dict[str, int],
    T: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    output_format: str = "channels_first",
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders.

    Args:
        train_files: List of training file paths
        test_files: List of test file paths
        label_map: Dict mapping folder names to class indices
        T: Fixed time series length
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        output_format: "channels_first" or "time_first"

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Build train dataset first (to lock feature_cols), then reuse for test
    train_ds = CycleDataset(train_files, label_map, T=T, output_format=output_format)
    test_ds = CycleDataset(
        test_files,
        label_map,
        T=T,
        feature_cols=train_ds.feature_cols,
        output_format=output_format,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def split_cycles_sorted_by_patient_type(
    df: pd.DataFrame,
    metadata_json_path: str,
    patients_json_path: str,
    out_dir: str = "data",
    unknown_folder: str = "UNKNOWN",
    analyte_value: str = "analyte",
) -> None:
    """Split cycles from a single CSV into separate files by patient type.

    This function processes a large CSV file containing multiple runs and cycles,
    and splits it into individual CSV files organized by patient type.

    Args:
        df: DataFrame with all cycle data
        metadata_json_path: Path to JSON mapping run_id -> patient_id
        patients_json_path: Path to JSON mapping patient_id -> type
        out_dir: Output directory for sorted CSV files
        unknown_folder: Folder name for patients with unknown type
        analyte_value: Section value to filter (keep only "analyte" rows)
    """
    os.makedirs(out_dir, exist_ok=True)

    def _clean(x: str) -> str:
        """Clean string for use in filename."""
        return str(x).replace("/", "-").replace("\\", "-").replace(" ", "_")

    # Load metadata: run_id -> patient_id
    with open(metadata_json_path, "r") as f:
        meta_list = json.load(f)
    run_to_patient = {
        str(m["run_id"]): str(m["patient_id"])
        for m in meta_list
        if "run_id" in m and "patient_id" in m
    }

    # Load patients: patient_id -> type
    with open(patients_json_path, "r") as f:
        patients_list = json.load(f)
    patient_to_type = {
        str(p["patient_id"]): str(p.get("type", unknown_folder)).strip()
        for p in patients_list
        if "patient_id" in p
    }

    df = df.copy()
    df["run_id"] = df["run_id"].astype(str)

    # Process each (run_id, cycle) group
    for (run_id, cycle), g in df.groupby(["run_id", "cycle"], sort=False):
        # Keep only analyte rows
        if "section" in g.columns:
            g = g[g["section"] == analyte_value]
        else:
            continue

        # Skip if no analyte rows
        if g.empty:
            continue

        # Get patient_id
        pid = run_to_patient.get(run_id)

        # Fallback: try df column if metadata missing
        if pid is None and "patient_id" in g.columns and not g["patient_id"].isna().all():
            pid = str(g["patient_id"].iloc[0])

        # Get patient type
        patient_type = patient_to_type.get(str(pid), unknown_folder)
        patient_type = _clean(patient_type) if patient_type else unknown_folder

        # Create type directory
        type_dir = os.path.join(out_dir, patient_type)
        os.makedirs(type_dir, exist_ok=True)

        # Save CSV
        fname = f"{_clean(run_id)}_{_clean(cycle)}.csv"
        g.to_csv(os.path.join(type_dir, fname), index=False)
