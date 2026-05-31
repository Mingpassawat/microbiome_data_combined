"""Shared helpers for one-vs-rest internal CV reporting."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold

CV_TYPE = "sample_stratified_kfold"
CV_ROLE = "internal_sanity_metric"
EXTERNAL_ROLE = "primary_generalization_metric"


def make_ovr_folds(
    y: np.ndarray,
    requested_folds: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Return valid sample-level stratified folds and metadata."""
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    meta = {
        "cv_type": CV_TYPE,
        "cv_role": CV_ROLE,
        "requested_folds": int(requested_folds),
        "actual_folds": 0,
        "cv_status": "unavailable",
        "cv_unavailable_reason": "",
    }
    if n_pos < 2 or n_neg < 2:
        meta["cv_unavailable_reason"] = (
            f"need at least 2 positives and 2 negatives; got pos={n_pos}, neg={n_neg}"
        )
        return [], meta

    n_splits = min(int(requested_folds), n_pos, n_neg)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(tr, va) for tr, va in splitter.split(np.zeros(len(y)), y)]
    for _, va_idx in folds:
        va_pos = int(y[va_idx].sum())
        va_neg = int((y[va_idx] == 0).sum())
        if va_pos == 0 or va_neg == 0:
            meta["cv_unavailable_reason"] = "stratified split produced a single-class fold"
            return [], meta

    meta["actual_folds"] = n_splits
    meta["cv_status"] = "available"
    meta.pop("cv_unavailable_reason")
    return folds, meta


def summarize_metric_dicts(
    rows: list[dict],
    exclude: Iterable[str] = ("n_pos", "n_neg"),
) -> dict:
    """Summarize metric dicts with nan-safe means/stds."""
    if not rows:
        return {"mean": {}, "std": {}, "folds": []}

    excluded = set(exclude)
    mean = {}
    std = {}
    for key in rows[0]:
        if key in excluded:
            continue
        values = np.array([r[key] for r in rows], dtype=float)
        if np.isnan(values).all():
            mean[key] = float("nan")
            std[key] = float("nan")
        else:
            mean[key] = float(np.nanmean(values))
            std[key] = float(np.nanstd(values))
    return {"mean": mean, "std": std, "folds": rows}


def external_validation_strength(task: dict) -> str:
    """Flag external validation sets built from same-study sample splits."""
    source = str(task.get("val_source", "none")).lower()
    if "sample split" in source:
        return "weak_sample_split_same_study"
    return "standard"


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if not math.isnan(float(v))]
