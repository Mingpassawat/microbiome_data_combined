#!/usr/bin/env python3
"""Example: load the combined cMD + GMHI + GMWI2 dataset for training.

This script intentionally uses only pandas and numpy. It gives you:

    X_train, y_train, X_val, y_val

for either a binary healthy-vs-nonhealthy task or a multiclass phenotype task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_combined_dataset(
    data_dir: str | Path = "data/combined_microbiome",
    task: str = "binary",
    dtype: str = "float32",
    matrix: str = "proportions",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[str]]:
    """Load combined microbiome features and labels.

    Args:
        data_dir: Folder produced by scripts/prepare_combined_microbiome_data.py.
        task: "binary" for healthy vs nonhealthy, or "multiclass" for phenotype labels.
        dtype: Numeric dtype for the feature matrix.
        matrix: "proportions" for 0-1 row-normalized input, or "raw" for source-scale audit data.

    Returns:
        X_train, y_train, X_val, y_val, label_to_id, feature_names.
    """
    data_dir = Path(data_dir)
    matrix_files = {
        "proportions": "relative_abundance_species_proportions_wide.csv",
        "raw": "relative_abundance_species_wide.csv",
    }
    if matrix not in matrix_files:
        raise ValueError("matrix must be 'proportions' or 'raw'")

    X = pd.read_csv(data_dir / matrix_files[matrix], index_col="sample_key")
    meta = pd.read_csv(data_dir / "metadata.csv", index_col="sample_key", low_memory=False)
    meta = meta.loc[X.index]

    if task == "binary":
        labels = np.where(meta["is_healthy"].astype(bool), "Healthy", "Nonhealthy")
    elif task == "multiclass":
        labels = meta["label"].astype(str).to_numpy()
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")

    label_names = sorted(pd.unique(labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = np.array([label_to_id[label] for label in labels], dtype=np.int64)

    split = meta["split"].astype(str).to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"

    feature_names = X.columns.to_list()
    values = X.to_numpy(dtype=dtype)
    X_train, y_train = values[train_mask], y[train_mask]
    X_val, y_val = values[val_mask], y[val_mask]

    return X_train, y_train, X_val, y_val, label_to_id, feature_names


def iter_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/combined_microbiome")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--matrix", choices=["proportions", "raw"], default="proportions")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, label_to_id, feature_names = load_combined_dataset(
        data_dir=args.data_dir,
        task=args.task,
        matrix=args.matrix,
    )

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, np.bincount(y_train).tolist())
    print("X_val:", X_val.shape, X_val.dtype)
    print("y_val:", y_val.shape, np.bincount(y_val).tolist())
    print("n_features:", len(feature_names))
    print("label_to_id:", json.dumps(label_to_id, indent=2))

    xb, yb = next(iter_minibatches(X_train, y_train, batch_size=args.batch_size))
    print("first batch:", xb.shape, yb.shape)


if __name__ == "__main__":
    main()
