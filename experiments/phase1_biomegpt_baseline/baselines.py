"""
Phase 1 — Classical ML baselines for the BiomeGPT comparison table.

Usage:
    python baselines.py

Runs RF, Logistic Regression, and XGBoost (if installed) on four feature
representations: raw proportions, log1p, CLR, and binned abundance.
Binary mode uses study-level GroupKFold CV + external val split.  OvR mode
uses sample-level shuffled StratifiedKFold as an internal sanity metric;
external validation remains the primary generalization metric.

Results saved to results/baseline_results.json.
"""
from __future__ import annotations

import argparse
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
from ovr_cv import (
    CV_TYPE,
    EXTERNAL_ROLE,
    external_validation_strength,
    make_ovr_folds,
    summarize_metric_dicts,
)


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
    return summarize_metric_dicts(folds, exclude=())


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
    n_folds: int,
    seed: int,
    disease: str,
) -> tuple[list[dict], dict]:
    """Sample-level stratified CV for a single binary OvR classifier."""
    folds, cv_meta = make_ovr_folds(y, n_folds, seed)
    if cv_meta["cv_status"] != "available":
        return [], cv_meta
    results = []
    fold_bar = tqdm(
        folds,
        total=cv_meta["actual_folds"],
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
    return results, cv_meta


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_classifiers(seed: int, n_pos: int, n_neg: int) -> dict:
    clfs: dict = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, solver="saga", penalty="l1", max_iter=2000,
                class_weight="balanced", random_state=seed, verbose=True,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", help="Run a single OvR disease task.")
    parser.add_argument("--classifier", help="Run a single classifier.")
    parser.add_argument("--feature", help="Run a single feature representation.")
    return parser.parse_args()


def _load_results(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_results(path: str, results: dict) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _is_complete(result: dict, require_ovr_cv_type: bool = False) -> bool:
    if not result:
        return False
    if require_ovr_cv_type:
        return result.get("cv", {}).get("cv_type") == CV_TYPE
    return "cv" in result and "external" in result


def main() -> None:
    args = _parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("task", {}).get("mode", "binary")

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
    if args.feature:
        if args.feature not in feature_sets:
            raise ValueError(f"Unknown feature: {args.feature}. Options: {sorted(feature_sets)}")
        active_features = [args.feature]

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
        if args.classifier:
            if args.classifier not in classifiers:
                raise ValueError(f"Unknown classifier: {args.classifier}. Options: {sorted(classifiers)}")
            classifiers = {args.classifier: classifiers[args.classifier]}
        loaded_results = _load_results(results_path)
        all_results: dict = {k: v for k, v in loaded_results.items() if k in feature_sets}
        combos = [(f, c) for f in active_features for c in classifiers]
        outer_bar = tqdm(combos, desc="Baselines", unit="run")
        for feat_name, clf_name in outer_bar:
            existing = all_results.get(feat_name, {}).get(clf_name, {})
            if _is_complete(existing):
                tqdm.write(f"\n{clf_name} | {feat_name} already complete — skip")
                continue
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
            all_results.setdefault(feat_name, {})[clf_name] = {
                "cv": cv_summary,
                "external": ext,
                "external_role": EXTERNAL_ROLE,
            }
            _save_results(results_path, all_results)
            tqdm.write(
                f"  CV  : acc={cv_summary['mean']['accuracy']:.4f}  "
                f"f1={cv_summary['mean']['macro_f1']:.4f}  "
                f"auroc={cv_summary['mean']['macro_auroc']:.4f}\n"
                f"  Ext : acc={ext['accuracy']:.4f}  "
                f"f1={ext['macro_f1']:.4f}  auroc={ext['macro_auroc']:.4f}"
            )
        tqdm.write(f"\nBaseline results saved → {results_path}")
        return

    # ── OvR mode: load precomputed splits, one classifier per disease ─────────
    splits_path = os.path.join(base_dir, cfg["data"]["task_splits_path"])
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"task_splits.json not found: {splits_path}\nRun make_task_splits.py first."
        )
    with open(splits_path) as f:
        task_splits: dict[str, dict] = json.load(f)
    ovr_tasks = {k: v for k, v in task_splits.items() if k != "binary"}
    if args.disease:
        if args.disease not in ovr_tasks:
            raise ValueError(f"Unknown disease: {args.disease}. Options: {sorted(ovr_tasks)}")
        ovr_tasks = {args.disease: ovr_tasks[args.disease]}
    tqdm.write(f"\nOvR tasks ({len(ovr_tasks)}): {list(ovr_tasks)}")

    # key → positional index in X_tr_raw / X_va_raw
    meta_with_label = _attach_label_column(meta, meta_path)
    train_meta = meta_with_label[meta_with_label["split"] == "train"]
    val_meta   = meta_with_label[meta_with_label["split"] == "val"]
    tr_key_to_pos = {k: i for i, k in enumerate(train_meta.index)}
    va_key_to_pos = {k: i for i, k in enumerate(val_meta.index)}

    def _gather_feat(keys: list[str], X_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
        tr_idx = [tr_key_to_pos[k] for k in keys if k in tr_key_to_pos]
        va_idx = [va_key_to_pos[k] for k in keys if k in va_key_to_pos]
        parts = []
        if tr_idx:
            parts.append(X_tr[tr_idx])
        if va_idx:
            parts.append(X_va[va_idx])
        if not parts:
            return np.empty((0, X_tr.shape[1]), dtype=X_tr.dtype)
        return np.concatenate(parts, axis=0)

    loaded_results = _load_results(results_path)
    all_ovr_results: dict = {k: v for k, v in loaded_results.items() if k in ovr_tasks}
    disease_bar = tqdm(ovr_tasks.items(), desc="OvR diseases", unit="disease", total=len(ovr_tasks))
    for disease, task in disease_bar:
        disease_bar.set_postfix(disease=disease[:30])

        tr_keys = [k for k in task["train_keys"] if k in tr_key_to_pos]
        if not tr_keys:
            tqdm.write(f"  [{disease}] no train keys — skip")
            continue
        tr_pos = np.array([tr_key_to_pos[k] for k in tr_keys])
        y_tr_d = np.array(
            [1 if meta_with_label.loc[k, "label"] == disease else 0 for k in tr_keys],
            dtype=int,
        )
        n_pos_d = int(y_tr_d.sum())
        n_neg_d = int((y_tr_d == 0).sum())
        if n_pos_d == 0 or n_neg_d == 0:
            tqdm.write(f"  [{disease}] degenerate (pos={n_pos_d} neg={n_neg_d}) — skip")
            continue

        ext_source = task.get("val_source", "none")
        classifiers_d = _make_classifiers(seed, n_pos_d, n_neg_d)
        if args.classifier:
            if args.classifier not in classifiers_d:
                raise ValueError(f"Unknown classifier: {args.classifier}. Options: {sorted(classifiers_d)}")
            classifiers_d = {args.classifier: classifiers_d[args.classifier]}
        combos = [(f, c) for f in active_features for c in classifiers_d]
        disease_results: dict = all_ovr_results.get(disease, {})

        for feat_name, clf_name in combos:
            existing = disease_results.get(feat_name, {}).get(clf_name, {})
            if _is_complete(existing, require_ovr_cv_type=True):
                tqdm.write(f"  [{disease}] {clf_name}/{feat_name} already complete — skip")
                continue
            X_tr_full, X_va_full = feature_sets[feat_name]
            X_tr_d = X_tr_full[tr_pos]
            clf = classifiers_d[clf_name]

            folds, cv_meta = run_ovr_cv(clf, X_tr_d, y_tr_d, n_folds, seed, disease)
            cv_summary = summarize_metric_dicts(folds)
            cv_summary.update(cv_meta)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr_d, y_tr_d)

            external: dict | None = None
            pos_keys = task["val_pos_keys"]
            neg_keys = task["val_neg_keys"]
            if pos_keys and neg_keys:
                X_pos = _gather_feat(pos_keys, X_tr_full, X_va_full)
                X_neg = _gather_feat(neg_keys, X_tr_full, X_va_full)
                if len(X_pos) > 0 and len(X_neg) > 0:
                    X_ext = np.concatenate([X_pos, X_neg], axis=0)
                    y_ext = np.concatenate([
                        np.ones(len(X_pos), dtype=int),
                        np.zeros(len(X_neg), dtype=int),
                    ])
                    y_prob = clf.predict_proba(X_ext)[:, 1]
                    y_pred = clf.predict(X_ext)
                    external = _binary_metrics(y_ext, y_pred, y_prob)

            disease_results.setdefault(feat_name, {})[clf_name] = {
                "cv": cv_summary,
                "external": external,
                "external_role": EXTERNAL_ROLE,
                "external_source": ext_source,
                "external_validation_strength": external_validation_strength(task),
                "n_train_pos": n_pos_d,
                "n_train_neg": n_neg_d,
            }
            all_ovr_results[disease] = disease_results
            _save_results(results_path, all_ovr_results)

        all_ovr_results[disease] = disease_results
        best = disease_results[active_features[0]]
        first_clf = next(iter(best))
        cv_mean = best[first_clf]["cv"]["mean"]
        ext = best[first_clf]["external"]
        ext_strength = best[first_clf].get("external_validation_strength", "standard")
        cv_auroc = cv_mean.get("auroc", float("nan"))
        cv_f1 = cv_mean.get("binary_f1", float("nan"))
        tqdm.write(
            f"  {disease[:35]:<35} | "
            f"internal CV auroc={cv_auroc:.4f}  f1={cv_f1:.4f}"
            + (f"  | ext auroc={ext['auroc']:.4f} ({ext_source}; {ext_strength})" if ext
               else f"  | no ext val ({ext_source})")
        )

    _save_results(results_path, all_ovr_results)
    tqdm.write(f"\nOvR baseline results saved → {results_path}")


if __name__ == "__main__":
    main()
