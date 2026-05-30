from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Bin 0 = absent species; bins 1..100 = nonzero abundance; 101 = [MASK]
MASK_BIN_ID: int = 101


def load_data(abundance_path: str, metadata_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align abundance matrix and metadata on sample_key index."""
    abun = pd.read_csv(abundance_path, index_col=0)
    meta = pd.read_csv(metadata_path, low_memory=False)

    # Metadata sometimes has a duplicate sample_key column
    if "sample_key.1" in meta.columns:
        meta = meta.drop(columns=["sample_key.1"])
    meta = meta.set_index("sample_key")
    meta = meta[~meta.index.duplicated(keep="first")]

    # Normalize is_healthy to int (handles bool strings from CSV)
    meta["is_healthy"] = (
        meta["is_healthy"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    )
    meta = meta.dropna(subset=["is_healthy"])
    meta["is_healthy"] = meta["is_healthy"].astype(int)

    # Fill missing group_id with study_id as fallback
    if "group_id" not in meta.columns:
        meta["group_id"] = meta.get("study_id", pd.Series(meta.index, index=meta.index))
    else:
        meta["group_id"] = meta["group_id"].fillna(meta.get("study_id", meta.index))

    common = meta.index.intersection(abun.index)
    meta = meta.loc[common]
    abun = abun.loc[common].astype(np.float32)
    return abun, meta


def compute_bin_edges(
    abun: pd.DataFrame, meta: pd.DataFrame, n_bins: int = 100
) -> np.ndarray:
    """
    Compute n_bins-1 interior quantile cut-points from train-split nonzero values.
    Must be computed only from train data; apply the same edges to val.
    Returns array of shape [n_bins-1].
    """
    train_idx = meta.index[meta["split"] == "train"]
    nz = abun.loc[train_idx].values.ravel()
    nz = nz[nz > 0]
    # n_bins-1 interior percentile points → n_bins equal-frequency bins
    pts = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(nz, pts)  # shape [n_bins-1]


def _digitize(values: np.ndarray, bin_edges: np.ndarray, n_bins: int) -> np.ndarray:
    """Map nonzero proportion values to integer bins 1..n_bins."""
    # searchsorted returns 0..n_bins-1 → +1 = 1..n_bins
    b = np.searchsorted(bin_edges, values, side="right") + 1
    return np.clip(b, 1, n_bins).astype(np.int32)


class MicrobiomeDataset(Dataset):
    """
    Sparse microbiome dataset. Each sample is represented as a list of
    (species_idx, bin_idx) pairs for its nonzero species only.

    Labels are read from `label_col` (default "is_healthy": 1=healthy, 0=diseased).
    For OvR fine-tuning, pass a pre-filtered meta with a custom binary label column.
    """

    def __init__(
        self,
        abun: pd.DataFrame,
        meta: pd.DataFrame,
        species_list: list[str],
        bin_edges: np.ndarray,
        split: str | None = None,
        n_bins: int = 100,
        label_col: str = "is_healthy",
    ) -> None:
        if split is not None:
            meta = meta[meta["split"] == split]
        common = meta.index.intersection(abun.index)
        meta = meta.loc[common]
        abun_vals = abun.loc[common].values  # [N, S] float32

        self.n_species = len(species_list)
        self.n_bins = n_bins

        self.species_idx_list: list[np.ndarray] = []
        self.bin_id_list: list[np.ndarray] = []
        self.labels: list[int] = []
        self.study_ids: list[str] = []
        self.sample_keys: list[str] = list(meta.index)

        for i in range(len(meta)):
            row = abun_vals[i]
            nz_idx = np.where(row > 0)[0].astype(np.int32)
            if len(nz_idx) == 0:
                # Edge case: treat as one species at bin 1 to avoid empty sequences
                nz_idx = np.zeros(1, dtype=np.int32)
                bins = np.ones(1, dtype=np.int32)
            else:
                bins = _digitize(row[nz_idx], bin_edges, n_bins)
            self.species_idx_list.append(nz_idx)
            self.bin_id_list.append(bins)
            self.labels.append(int(meta[label_col].iloc[i]))
            self.study_ids.append(str(meta["group_id"].iloc[i]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.species_idx_list[idx].copy()).long(),
            torch.from_numpy(self.bin_id_list[idx].copy()).long(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_species: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepend [CLS] token (species_idx=n_species) and right-pad to the maximum
    sequence length in the batch.

    key_padding_mask: True = padding position (ignored by transformer).
    """
    sp_list, bin_list, labels = zip(*batch)

    cls_id = n_species
    max_len = max(len(s) for s in sp_list) + 1  # +1 for [CLS]
    B = len(batch)

    species_ids = torch.zeros(B, max_len, dtype=torch.long)
    bin_ids = torch.zeros(B, max_len, dtype=torch.long)
    # True = ignore (padding); initialise all as padding, then mark real tokens
    key_padding_mask = torch.ones(B, max_len, dtype=torch.bool)

    for i, (sp, bi) in enumerate(zip(sp_list, bin_list)):
        L = len(sp) + 1
        species_ids[i, 0] = cls_id
        species_ids[i, 1:L] = sp
        bin_ids[i, 1:L] = bi
        key_padding_mask[i, :L] = False  # valid tokens

    return species_ids, bin_ids, key_padding_mask, torch.stack(labels)


def make_collate_fn(n_species: int):
    return partial(collate_fn, n_species=n_species)
