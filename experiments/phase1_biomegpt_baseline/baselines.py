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
import pandas as pd
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

def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    return {"binary_f1": float(f1), "auroc": float(auroc)}


def _build_disease_list(
    meta: pd.DataFrame,
    min_samples: int,
    focus: list[str] | None = None,
) -> list[str]:
    train_meta = meta[meta["split"] == "train"]
    counts = train_meta["label"].value_counts()
    diseases = [
        label
        for label, n in counts.items()
        if label != "Healthy" and ";" not in str(label) and n >= min_samples
    ]
    if focus:
        found = [d for d in diseases if d in focus]
        missing = set(focus) - set(found)
        if missing:
            tqdm.write(
                f"  Warning: focus diseases not in train data or < {min_samples} samples: {sorted(missing)}"
            )
        diseases = found
    return diseases


def _attach_label_column(meta: pd.DataFrame, meta_path: str) -> pd.DataFrame:
    """Read raw 'label' column from disk and attach it to meta."""
    raw = pd.read_csv(meta_path, low_memory=False)
    if "sample_key.1" in raw.columns:
        raw = raw.drop(columns=["sample_key.1"])
    raw = raw.set_index("sample_key")
    raw = raw[~raw.index.duplicated(keep="first")]
    meta = meta.copy()
    meta["label"] = raw["label"].reindex(meta.index)
    return meta


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


def run_ovr_cv(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    disease: str,
) -> list[dict]:
    """GroupKFold CV for a single binary OvR classifier."""
    n_splits = min(n_folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    fold_bar = tqdm(
        gkf.split(X, y, groups),
        total=n_splits,
        desc=f"  [{disease[:20]}] CV",
        unit="fold",
        leave=False,
    )
    for tr_idx, va_idx in fold_bar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr_idx], y[tr_idx])
        y_prob = clf.predict_proba(X[va_idx])[:, 1]
        y_pred = clf.predict(X[va_idx])
        m = _binary_metrics(y[va_idx], y_pred, y_prob)
        results.append(m)
        fold_bar.set_postfix(auroc=f"{m['auroc']:.4f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_classifiers(seed: int, n_pos: int, n_neg: int) -> dict:
    clfs: dict = {
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
        clfs["xgboost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=n_neg / max(n_pos, 1),
            subsample=0.8, colsample_bytree=0.5,
            eval_metric="logloss", verbosity=0,
            random_state=seed, n_jobs=-1,
        )
    return clfs


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    task_cfg = cfg.get("task", {})
    mode = task_cfg.get("mode", "binary")
    focus_diseases: list[str] = task_cfg.get("focus_diseases") or []

    set_seed(cfg["baselines"]["seed"])
    seed = cfg["baselines"]["seed"]
    n_folds = cfg["baselines"]["n_cv_folds"]
    n_bins = cfg["data"]["n_bins"]

    abun_path = os.path.join(base_dir, cfg["data"]["abundance_path"])
    meta_path = os.path.join(base_dir, cfg["data"]["metadata_path"])
    tqdm.write(f"Loading data…  mode={mode}")
    abun, meta = load_data(abun_path, meta_path)
    bin_edges = compute_bin_edges(abun, meta, n_bins)

    train_meta = meta[meta["split"] == "train"]
    val_meta = meta[meta["split"] == "val"]
    X_tr_raw = abun.loc[train_meta.index].values.astype(np.float32)
    X_va_raw = abun.loc[val_meta.index].values.astype(np.float32)

    # Feature representations (computed once, shared across modes)
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "proportions": (X_tr_raw, X_va_raw),
        "log1p":       (log1p_transform(X_tr_raw), log1p_transform(X_va_raw)),
        "clr":         (clr_transform(X_tr_raw), clr_transform(X_va_raw)),
        "binned":      (
            bin_transform(X_tr_raw, bin_edges, n_bins),
            bin_transform(X_va_raw, bin_edges, n_bins),
        ),
    }
    # Change this list to run more/fewer feature × classifier combos
    active_features = ["log1p"]

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(base_dir, cfg["baselines"]["results_path"])

    # ── Binary mode: healthy vs. all-diseased ────────────────────────────────
    if mode == "binary":
        y_tr = train_meta["is_healthy"].values.astype(int)
        y_va = val_meta["is_healthy"].values.astype(int)
        groups = train_meta["group_id"].values
        pos = int(y_tr.sum())
        neg = int((y_tr == 0).sum())
        tqdm.write(
            f"Train: {len(y_tr)} samples ({pos} healthy / {neg} diseased)\n"
            f"Val:   {len(y_va)} samples ({y_va.sum()} healthy / {(1-y_va).sum()} diseased)"
        )

        classifiers = _make_classifiers(seed, pos, neg)
        all_results: dict = {}
        combos = [(f, c) for f in active_features for c in classifiers]
        outer_bar = tqdm(combos, desc="Baselines", unit="run")
        for feat_name, clf_name in outer_bar:
            X_tr, X_va = feature_sets[feat_name]
            clf = classifiers[clf_name]
            outer_bar.set_description(f"{clf_name} | {feat_name}")
            tqdm.write(f"\n{clf_name} | {feat_name} — {n_folds}-fold CV…")
            folds = run_cv(clf, X_tr, y_tr, groups, n_folds, desc=f"{clf_name}/{feat_name}")
            cv_summary = _mean_std(folds)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr, y_tr)
            y_prob_va = clf.predict_proba(X_va)[:, 1]
            y_pred_va = clf.predict(X_va)
            ext = _metrics(y_va, y_pred_va, y_prob_va)
            all_results.setdefault(feat_name, {})[clf_name] = {"cv": cv_summary, "external": ext}
            tqdm.write(
                f"  CV  : acc={cv_summary['mean']['accuracy']:.4f}  "
                f"f1={cv_summary['mean']['macro_f1']:.4f}  "
                f"auroc={cv_summary['mean']['macro_auroc']:.4f}\n"
                f"  Ext : acc={ext['accuracy']:.4f}  "
                f"f1={ext['macro_f1']:.4f}  auroc={ext['macro_auroc']:.4f}"
            )

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        tqdm.write(f"\nBaseline results saved → {results_path}")
        return

    # ── OvR mode: one classifier per disease vs. healthy ─────────────────────
    meta = _attach_label_column(meta, meta_path)
    train_meta = meta[meta["split"] == "train"]
    val_meta = meta[meta["split"] == "val"]

    min_samples = cfg["finetune_ovr"]["min_disease_samples"]
    diseases = _build_disease_list(
        meta,
        min_samples,
        focus=focus_diseases if mode == "focused_ovr" else None,
    )
    tqdm.write(f"\nOvR diseases ({len(diseases)}): {diseases}")

    # Positional index maps for fast row selection
    tr_key_to_pos = {k: i for i, k in enumerate(train_meta.index)}
    va_key_to_pos = {k: i for i, k in enumerate(val_meta.index)}

    # Rebuild classifiers with balanced weights (no pos/neg ratio yet — set per disease)
    all_ovr_results: dict = {}
    disease_bar = tqdm(diseases, desc="OvR diseases", unit="disease")
    for disease in disease_bar:
        disease_bar.set_postfix(disease=disease[:30])

        # Train subset: disease samples + healthy samples
        tr_mask = (train_meta["label"] == disease) | (train_meta["label"] == "Healthy")
        dm = train_meta[tr_mask].copy()
        dm["_ovr"] = (dm["label"] == disease).astype(int)
        tr_keys = [k for k in dm.index if k in tr_key_to_pos]
        if not tr_keys:
            tqdm.write(f"  [{disease}] no train samples — skip")
            continue
        tr_pos = np.array([tr_key_to_pos[k] for k in tr_keys])
        y_tr_d = dm.loc[tr_keys, "_ovr"].values.astype(int)
        groups_d = dm.loc[tr_keys, "group_id"].values
        n_pos_d = int(y_tr_d.sum())
        n_neg_d = int((y_tr_d == 0).sum())
        if n_pos_d == 0 or n_neg_d == 0:
            tqdm.write(f"  [{disease}] degenerate (pos={n_pos_d} neg={n_neg_d}) — skip")
            continue

        # Val subset
        va_mask = (val_meta["label"] == disease) | (val_meta["label"] == "Healthy")
        vm = val_meta[va_mask].copy()
        vm["_ovr"] = (vm["label"] == disease).astype(int)
        va_keys = [k for k in vm.index if k in va_key_to_pos]
        va_pos = np.array([va_key_to_pos[k] for k in va_keys]) if va_keys else np.array([], dtype=int)
        y_va_d = vm.loc[va_keys, "_ovr"].values.astype(int) if va_keys else np.array([], dtype=int)

        classifiers_d = _make_classifiers(seed, n_pos_d, n_neg_d)
        combos = [(f, c) for f in active_features for c in classifiers_d]
        disease_results: dict = {}

        for feat_name, clf_name in combos:
            X_tr_full, X_va_full = feature_sets[feat_name]
            X_tr_d = X_tr_full[tr_pos]
            clf = classifiers_d[clf_name]

            folds = run_ovr_cv(clf, X_tr_d, y_tr_d, groups_d, n_folds, disease)
            cv_summary = _mean_std(folds)

            # Train final, eval on val
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr_d, y_tr_d)
            external: dict | None = None
            if len(va_pos) > 0 and y_va_d.sum() > 0 and (y_va_d == 0).sum() > 0:
                X_va_d = X_va_full[va_pos]
                y_prob_va = clf.predict_proba(X_va_d)[:, 1]
                y_pred_va = clf.predict(X_va_d)
                external = _binary_metrics(y_va_d, y_pred_va, y_prob_va)

            disease_results.setdefault(feat_name, {})[clf_name] = {
                "cv": cv_summary,
                "external": external,
                "n_train_pos": n_pos_d,
                "n_train_neg": n_neg_d,
            }

        all_ovr_results[disease] = disease_results
        best = disease_results[active_features[0]]
        first_clf = next(iter(best))
        cv_mean = best[first_clf]["cv"]["mean"]
        ext = best[first_clf]["external"]
        tqdm.write(
            f"  {disease[:35]:<35} | "
            f"CV auroc={cv_mean['auroc']:.4f}  f1={cv_mean['binary_f1']:.4f}"
            + (f"  | ext auroc={ext['auroc']:.4f}" if ext else "  | no ext val")
        )

    with open(results_path, "w") as f:
        json.dump(all_ovr_results, f, indent=2)
    tqdm.write(f"\nOvR baseline results saved → {results_path}")


if __name__ == "__main__":
    main()
