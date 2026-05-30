"""
Phase 1 — Classical ML baselines for the BiomeGPT comparison table.

Usage:
    python baselines.py

Runs RF, Logistic Regression, and XGBoost (if installed) on four feature
representations: raw proportions, log1p, CLR, and binned abundance.
Evaluates with 10-fold study-level GroupKFold CV + external val split.

Results saved to results/baseline_results.json.
"""
from __future__ import annotations

import json
import os
import random
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import yaml

try:
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("xgboost not installed — skipping XGBoost baselines.")

from dataset import compute_bin_edges, load_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ── Feature transforms ────────────────────────────────────────────────────────

def clr_transform(X: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform. Zeros replaced with pseudocount."""
    X_ps = X + pseudocount
    log_X = np.log(X_ps)
    return log_X - log_X.mean(axis=1, keepdims=True)


def log1p_transform(X: np.ndarray) -> np.ndarray:
    return np.log1p(X)


def bin_transform(
    X: np.ndarray, bin_edges: np.ndarray, n_bins: int = 100
) -> np.ndarray:
    """Map proportion matrix to integer bin values 0..n_bins."""
    out = np.zeros(X.shape, dtype=np.float32)
    nz = X > 0
    if nz.any():
        b = np.searchsorted(bin_edges, X[nz], side="right") + 1
        out[nz] = np.clip(b, 1, n_bins)
    return out


# ── Metric helpers ────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    return {"accuracy": float(acc), "macro_f1": float(f1), "macro_auroc": float(auroc)}


def _mean_std(folds: list[dict]) -> dict:
    mean = {k: float(np.mean([r[k] for r in folds])) for k in folds[0]}
    std = {k: float(np.std([r[k] for r in folds])) for k in folds[0]}
    return {"mean": mean, "std": std, "folds": folds}


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_cv(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    desc: str = "CV",
) -> list[dict]:
    gkf = GroupKFold(n_splits=n_folds)
    results = []
    fold_bar = tqdm(gkf.split(X, y, groups), total=n_folds, desc=f"  {desc}", unit="fold", leave=True)
    for tr_idx, va_idx in fold_bar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr_idx], y[tr_idx])
        y_prob = clf.predict_proba(X[va_idx])[:, 1]
        y_pred = clf.predict(X[va_idx])
        m = _metrics(y[va_idx], y_pred, y_prob)
        results.append(m)
        fold_bar.set_postfix(auroc=f"{m['macro_auroc']:.4f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["baselines"]["seed"])
    seed = cfg["baselines"]["seed"]
    n_folds = cfg["baselines"]["n_cv_folds"]
    n_bins = cfg["data"]["n_bins"]

    abun_path = os.path.join(base_dir, cfg["data"]["abundance_path"])
    meta_path = os.path.join(base_dir, cfg["data"]["metadata_path"])
    print("Loading data…")
    abun, meta = load_data(abun_path, meta_path)

    bin_edges = compute_bin_edges(abun, meta, n_bins)

    train_meta = meta[meta["split"] == "train"]
    val_meta = meta[meta["split"] == "val"]

    X_tr_raw = abun.loc[train_meta.index].values.astype(np.float32)
    X_va_raw = abun.loc[val_meta.index].values.astype(np.float32)
    y_tr = train_meta["is_healthy"].values.astype(int)
    y_va = val_meta["is_healthy"].values.astype(int)
    groups = train_meta["group_id"].values

    print(
        f"Train: {len(y_tr)} samples ({y_tr.sum()} healthy / {(1-y_tr).sum()} diseased)"
    )
    print(
        f"Val:   {len(y_va)} samples ({y_va.sum()} healthy / {(1-y_va).sum()} diseased)"
    )

    # Feature representations
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "proportions": (X_tr_raw, X_va_raw),
        "log1p":       (log1p_transform(X_tr_raw), log1p_transform(X_va_raw)),
        "clr":         (clr_transform(X_tr_raw), clr_transform(X_va_raw)),
        "binned":      (
            bin_transform(X_tr_raw, bin_edges, n_bins),
            bin_transform(X_va_raw, bin_edges, n_bins),
        ),
    }

    # Classifiers
    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())

    classifiers: dict[str, object] = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, solver="saga", penalty="l1", max_iter=2000,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
    }
    if _HAS_XGB:
        classifiers["xgboost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=neg / max(pos, 1),
            subsample=0.8, colsample_bytree=0.5,
            eval_metric="logloss", verbosity=0,
            random_state=seed, n_jobs=-1,
        )

    all_results: dict = {}

    # combos = [(f, c) for f in feature_sets for c in classifiers]
    combos = [(f, c) for f in ['log1p'] for c in classifiers]
    outer_bar = tqdm(combos, desc="Baselines", unit="run")
    for feat_name, clf_name in outer_bar:
        X_tr, X_va = feature_sets[feat_name]
        clf = classifiers[clf_name]
        outer_bar.set_description(f"{clf_name} | {feat_name}")
        tqdm.write(f"\n[{combos.index((feat_name, clf_name))+1}/{len(combos)}] {clf_name} | {feat_name} — starting {n_folds}-fold CV…")
        all_results.setdefault(feat_name, {})
        folds = run_cv(clf, X_tr, y_tr, groups, n_folds, desc=f"{clf_name}/{feat_name}")
        cv_summary = _mean_std(folds)

        # Train on full train set, evaluate on external val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
        y_prob_va = clf.predict_proba(X_va)[:, 1]
        y_pred_va = clf.predict(X_va)
        ext = _metrics(y_va, y_pred_va, y_prob_va)

        all_results[feat_name][clf_name] = {"cv": cv_summary, "external": ext}
        tqdm.write(
            f"\n{clf_name} | {feat_name}\n"
            f"  CV  : acc={cv_summary['mean']['accuracy']:.4f}  "
            f"f1={cv_summary['mean']['macro_f1']:.4f}  "
            f"auroc={cv_summary['mean']['macro_auroc']:.4f}\n"
            f"  Ext : acc={ext['accuracy']:.4f}  "
            f"f1={ext['macro_f1']:.4f}  auroc={ext['macro_auroc']:.4f}"
        )

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(base_dir, cfg["baselines"]["results_path"])
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBaseline results saved → {results_path}")


if __name__ == "__main__":
    main()
