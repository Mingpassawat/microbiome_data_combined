# Baselines — Step-by-Step

`baselines.py` answers one question: **how well do classical ML models do on the same data and same split as BiomeGPT?** If the transformer only barely beats RF, the transformer is not earning its complexity. If it wins clearly, especially on external val, the pretraining is doing real work.

The script has two operating modes controlled by `task.mode` in `config.yaml`:

| `task.mode`   | What runs                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------- |
| `binary`      | One classifier per feature set — healthy (1) vs. all-diseased (0)                           |
| `focused_ovr` | One classifier per disease listed in `task.focus_diseases` — disease (1) vs. healthy (0)    |
| `all_ovr`     | Same as `focused_ovr` but discovers all diseases with ≥ `min_disease_samples` train samples |

Steps 1–6 below describe the **binary mode**. Steps 1–3 are shared with OvR mode. The OvR-specific loop is described in [OvR Mode](#ovr-mode-focused_ovr--all_ovr).

---

## Step 1 — Load data and compute bin edges

```python
abun, meta = load_data(abun_path, meta_path)
bin_edges  = compute_bin_edges(abun, meta, n_bins=100)
```

`load_data` gives two aligned DataFrames:

- `abun`: shape `[33442, 2817]` — relative abundance proportions, values 0–1
- `meta`: one row per sample with `split`, `is_healthy`, `group_id`

`compute_bin_edges` computes 99 quantile cut-points **from train-split nonzero values only**. These are the same boundaries the transformer uses. They are needed here so the `binned` feature set is constructed identically.

---

## Step 2 — Split into train and val arrays

```python
train_meta = meta[meta["split"] == "train"]   # 30,711 rows
val_meta   = meta[meta["split"] == "val"]     #  2,731 rows

X_tr_raw = abun.loc[train_meta.index].values  # [30711, 2817]  float32
X_va_raw = abun.loc[val_meta.index].values    # [ 2731, 2817]  float32
y_tr     = train_meta["is_healthy"].values    # [30711]  int  1=healthy 0=diseased
y_va     = val_meta["is_healthy"].values      # [ 2731]  int
groups   = train_meta["group_id"].values      # [30711]  str  study identifiers for CV
```

`X_tr_raw` and `X_va_raw` are dense numpy matrices (each row is all 2,817 species values for one sample, including zeros). They are the starting point for every feature transform.

The val split is held out completely until the very end. It is never used to fit any transform or classifier.

---

## Step 3 — Build four feature representations

The same raw abundance matrix is transformed four different ways. All transforms are applied to both train and val using only parameters computed from train.

### 3a — `proportions` (raw, no transform)

```python
X_tr = X_tr_raw   # [30711, 2817]  values 0..1
X_va = X_va_raw
```

Values are already relative abundances (each row sums to ≤ 1). The simplest possible input. Most values are exactly 0.

### 3b — `log1p`

```python
X_tr = log(1 + X_tr_raw)
X_va = log(1 + X_va_raw)
```

Compresses large abundance values and keeps zeros at exactly 0 (because `log(1+0)=0`). Makes the distribution less right-skewed without needing a pseudocount. Applied independently to every cell — no parameters to fit, no train/val dependency.

### 3c — `clr` (centered log-ratio)

```python
X_ps  = X_tr_raw + 1e-6          # add tiny pseudocount to handle zeros
log_X = log(X_ps)                # log of each cell
X_tr  = log_X - mean(log_X, per_row)   # subtract per-sample geometric mean
```

The standard transform for compositional data. The idea: because proportions must sum to 1, a high value in one species mathematically forces lower values elsewhere. CLR removes this coupling by expressing each species relative to the geometric mean of all species in that sample. The pseudocount `1e-6` prevents `log(0)`.

Applied separately to train and val — no fitting needed, each row is transformed independently.

### 3d — `binned`

```python
# For each nonzero value in X_tr_raw:
bin_id = searchsorted(bin_edges, value, side='right') + 1   # → 1..100
# Zero stays 0.
```

Maps each abundance to the same integer bins the transformer uses. This lets you test whether the binning step alone (without the transformer) adds predictive value when fed into a classical model. The `bin_edges` are the quantile cut-points computed in Step 1 from train data only, then applied to val.

---

## Step 4 — Define classifiers

Classifiers are created by `_make_classifiers(seed, n_pos, n_neg)`, which takes the positive and negative class counts so XGBoost's `scale_pos_weight` can be set correctly per task. In binary mode they are built once; in OvR mode they are rebuilt per disease since each has a different imbalance ratio.

### Logistic Regression

```python
Pipeline([
    StandardScaler(),           # z-score each feature: (x - mean) / std
    LogisticRegression(
        penalty='l1',           # L1 regularization → sparse solution (many coefficients go to zero)
        solver='saga',          # only solver that supports L1 + large datasets
        C=1.0,                  # inverse regularization strength (smaller = more regularization)
        class_weight='balanced',# upweight minority class automatically
        max_iter=2000,
    )
])
```

`StandardScaler` is fitted on the fold's training rows only, then applied to the fold's val rows. This prevents val statistics from leaking into training.

L1 regularization produces a sparse model — most of the 2,817 species coefficients will be pushed to exactly zero. The model effectively performs feature selection automatically.

`class_weight='balanced'` sets each class's weight to `n_samples / (n_classes × n_class_samples)`. In OvR mode the minority class (disease) is typically much smaller than healthy, so this is important.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=300,        # 300 decision trees in the ensemble
    max_features='sqrt',     # each split considers sqrt(2817) ≈ 53 features at random
    class_weight='balanced',
)
```

Each tree is trained on a bootstrap sample of the training rows. At every node split, only a random subset of 53 features is considered. The 300 trees vote by majority for classification and average for probability.

`max_features='sqrt'` is the standard setting for classification — it adds enough randomness that the trees are decorrelated from each other, which is what makes the ensemble better than any single tree.

No feature scaling needed — decision trees are invariant to monotone transforms.

### XGBoost (if installed)

```python
XGBClassifier(
    n_estimators=300,        # 300 boosting rounds (sequential trees)
    max_depth=6,             # maximum depth of each tree
    learning_rate=0.1,       # shrinkage — each tree's contribution is scaled down
    scale_pos_weight=n_neg/n_pos,  # compensates for class imbalance; recalculated per disease in OvR
    subsample=0.8,           # each tree trained on 80% of rows (row subsampling)
    colsample_bytree=0.5,    # each tree uses 50% of features (column subsampling)
)
```

Unlike RF (parallel independent trees), XGBoost builds trees sequentially: each tree corrects the residual errors of all previous trees. `learning_rate=0.1` prevents any single tree from dominating.

`subsample` and `colsample_bytree` add regularization by introducing randomness into each tree.

---

## Step 5 — Run 10-fold study-level GroupKFold CV

This is the inner loop that runs for every `(feature_set, classifier)` combination. By default only `log1p` is active (`active_features = ["log1p"]`), giving 2–3 runs (LR + RF + optional XGBoost). Change `active_features` in the script to run all four feature representations.

```python
gkf = GroupKFold(n_splits=10)
for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups):
    clf.fit(X_tr[tr_idx], y_tr[tr_idx])
    y_prob = clf.predict_proba(X_tr[va_idx])[:, 1]  # probability of healthy
    y_pred = clf.predict(X_tr[va_idx])
    # compute accuracy, macro F1, macro AUROC
```

`GroupKFold` ensures no study appears in both the fold train and fold val. With 222 unique `group_id` values and 10 folds, each fold holds out roughly 22 studies (~3,000 samples) and trains on the remaining 200 studies (~27,700 samples).

This matters because: if the same study (same cohort, same lab, same sequencing machine) appears in both train and val, the model benefits from study-specific batch effects rather than generalizing to new cohorts. GroupKFold removes that shortcut.

`predict_proba(...)[:, 1]` — index `1` is the probability of the positive class (healthy = 1). This is used for AUROC, which requires a continuous score rather than a hard prediction.

After 10 folds, the results are averaged:

```python
cv_mean = { "accuracy": mean over 10 folds,
            "macro_f1": mean over 10 folds,
            "macro_auroc": mean over 10 folds }
cv_std  = { same keys, std over 10 folds }
```

---

## Step 6 — External validation

After CV, retrain the classifier on all 30,711 train samples and evaluate on the 2,731 val-split samples:

```python
clf.fit(X_tr, y_tr)                       # full train set
y_prob_va = clf.predict_proba(X_va)[:, 1]
y_pred_va = clf.predict(X_va)
ext = { accuracy, macro_f1, macro_auroc }
```

These 14 val-split studies were never seen during CV or any feature fitting. This is the closest thing to a real deployment test — train on existing cohorts, predict on entirely new ones.

---

## Step 7 — Compute metrics

Three metrics are computed for both CV and external val:

| Metric          | Formula               | Why                                                          |
| --------------- | --------------------- | ------------------------------------------------------------ |
| **Accuracy**    | correct / total       | Interpretable but misleading with class imbalance            |
| **Macro F1**    | mean of F1 per class  | Treats healthy and diseased equally regardless of class size |
| **Macro AUROC** | mean of AUC per class | Threshold-free; measures ranking quality; primary metric     |

With ~64% healthy and ~36% diseased, accuracy alone would be 64% for a model that always predicts healthy. Macro AUROC is the number to watch.

`macro_auroc` = `roc_auc_score(y_true, y_prob)` for binary classification, which is identical to the standard AUROC. For multi-class it would be the average over one-vs-rest AUROCs, but here the binary case applies.

---

## Step 8 — Save results

### Binary mode result format

```
results/baseline_results.json
{
  "log1p": {
    "logistic_regression": {
      "cv":       { "mean": {"accuracy": ..., "macro_f1": ..., "macro_auroc": ...},
                    "std":  {...},
                    "folds": [...] },
      "external": { "accuracy": ..., "macro_f1": ..., "macro_auroc": ... }
    },
    "random_forest": { ... },
    "xgboost":       { ... }
  }
}
```

### OvR mode result format

```
results/baseline_results.json
{
  "CRC": {
    "log1p": {
      "logistic_regression": {
        "cv":          { "mean": {"binary_f1": ..., "auroc": ...}, "std": {...}, "folds": [...] },
        "external":    { "binary_f1": ..., "auroc": ... },   # null if disease absent in val
        "n_train_pos": 312,
        "n_train_neg": 21444
      },
      "random_forest": { ... }
    }
  },
  "IBD": { ... },
  ...
}
```

The full fold-by-fold breakdown is under `"folds"` in both formats.

---

## OvR Mode (`focused_ovr` / `all_ovr`)

When `task.mode` is not `binary`, the script runs one classifier per disease instead of the single healthy-vs-all-diseased classifier.

### Disease discovery

```python
_build_disease_list(meta, min_samples=50, focus=focus_diseases)
```

Reads the raw `label` column (loaded via `_attach_label_column`). Excludes `"Healthy"` and composite labels containing `";"`. Keeps diseases with ≥ `min_disease_samples` train samples. In `focused_ovr` mode, further filters to only the diseases listed in `task.focus_diseases`; prints a warning for any listed disease not found or below threshold.

### Per-disease subset

For each disease:

```python
# Train: disease samples + healthy samples
train_mask = (train_meta["label"] == disease) | (train_meta["label"] == "Healthy")
y_tr_d = 1 if label == disease else 0   # binary OvR label
```

Row selection uses a precomputed `{sample_key → positional index}` map to index into the pre-built feature matrices without recomputing transforms.

### CV and external val

Uses `run_ovr_cv` with sample-level shuffled `StratifiedKFold` (`n_splits = min(n_folds, n_pos, n_neg)`). Reports `binary_f1` and `auroc` per fold. This OvR CV is an internal sanity/checkpoint metric, not a study-generalization claim; external validation remains the primary generalization metric.

### External val source per disease

| Disease           | External val source                                                    |
| ----------------- | ---------------------------------------------------------------------- |
| Colorectal cancer | val split (202 disease samples)                                        |
| IBD               | val split (204 disease samples)                                        |
| Obesity           | holdout: `gmhi:V-12_Obesity` (104 samples) + val-split Healthy         |
| Type 2 diabetes   | holdout: `cmd:MetaCardis_2020_a` (549 samples) + val-split Healthy     |
| Liver Cirrhosis   | weak sample split within the single positive study + val-split Healthy |

For holdout diseases: the holdout study is excluded from training; its disease samples + all val-split Healthy samples form the external test set. `external_source` is stored in the JSON output per result entry.

### Comparison table

After running both `finetune_ovr.py` and `baselines.py` in OvR mode, compare:

```
disease             | baseline CV auroc | transformer CV auroc | baseline ext           | transformer ext
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Colorectal cancer   |        ?          |          ?           | ? (val split)          | ? (val split)
IBD                 |        ?          |          ?           | ? (val split)          | ? (val split)
Obesity             |        ?          |          ?           | ? (holdout study)      | ? (holdout study)
Type 2 diabetes     |        ?          |          ?           | ? (holdout study)      | ? (holdout study)
Liver Cirrhosis     |        ?          |          ?           | n/a (1 study only)     | n/a (1 study only)
```

A transformer that only matches RF on CV but wins on external is still earning its pretraining.

---

## What the results tell you

Read the table like this:

```
feature       classifier     CV auroc   Ext auroc
──────────────────────────────────────────────────
proportions   logistic_reg    ?          ?
log1p         logistic_reg    ?          ?
clr           logistic_reg    ?          ?
binned        logistic_reg    ?          ?
proportions   random_forest   ?          ?
...
BiomeGPT                     91.7%      81.0%   (target)
```

Key comparisons:

- **CV vs. Ext gap**: A large gap (e.g., 90% CV → 60% ext) means the model is fitting study-specific artifacts. This should be similar for both classicals and the transformer.
- **Binned vs. other features**: If `binned` beats `proportions` for the baselines, the discretization itself helps and is not just a modeling artifact of the transformer.
- **BiomeGPT vs. best baseline**: The transformer should beat classicals on external val, especially AUROC. If it doesn't, the pretraining is not learning transferable representations.
