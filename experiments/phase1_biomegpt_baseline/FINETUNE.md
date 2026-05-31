# Fine-tuning — Step-by-Step

Two scripts implement fine-tuning on top of the frozen BiomeGPT encoder:

| Script            | Task                                     | Output                              |
| ----------------- | ---------------------------------------- | ----------------------------------- |
| `finetune.py`     | Binary: healthy (1) vs. all-diseased (0) | `results/finetune_results.json`     |
| `finetune_ovr.py` | OvR: one binary classifier per disease   | `results/finetune_ovr_results.json` |

Both share the same pretrained encoder and the same `[CLS]` embedding cache (`results/cls_emb_epoch{N}.pt`). Run `finetune.py` first — it writes the cache; `finetune_ovr.py` reuses it.

The active experiment scope (`binary` / `all_ovr` / `focused_ovr`) is set via `task.mode` in `config.yaml`. `finetune.py` is unaffected by this flag; `finetune_ovr.py` exits early if `mode = binary`.

---

## Shared Steps (both scripts)

### Step 1 — Load pretrain checkpoint

```python
ckpt = torch.load("results/pretrain_checkpoint.pt")
species_list = ckpt["species_list"]  # vocabulary of 2,817 species
bin_edges    = ckpt["bin_edges"]     # 99 quantile cut-points from train data
```

The checkpoint bundles the species vocabulary and bin edges alongside the model weights so the tokenization is always consistent with what the encoder was trained on. Using different bin edges at fine-tuning time would change the meaning of every token.

### Step 2 — Rebuild and freeze the encoder

```python
encoder = BiomeGPT(n_species=2817, d_model=512, n_heads=8, n_layers=8, ...)
encoder.load_state_dict(ckpt["model_state"])
encoder.eval()   # disables dropout
# No requires_grad_(False) needed — encoder is only called inside torch.no_grad()
```

The encoder is never updated during fine-tuning. Freezing it keeps the learned microbiome representations intact and makes fine-tuning fast — training is entirely on the small MLP head.

### Step 3 — Extract [CLS] embeddings (single pass)

```python
# One forward pass over all N samples
for batch in DataLoader(dataset, ...):
    emb = encoder.get_cls_embeddings(species_ids, bin_ids, pad_mask)  # [B, 512]
    all_emb.append(emb.cpu())
# Result: train_emb [30711, 512], val_emb [2731, 512]
```

Because the encoder is frozen and in `eval()` mode (no dropout), every forward pass produces the same output. Computing embeddings once and caching them is therefore **exactly equivalent** to running the full encoder every epoch — but ~50× faster for 50 training epochs.

The cache is saved to `results/cls_emb_epoch{N}.pt`. Both `finetune.py` and `finetune_ovr.py` check for this file and skip extraction if it exists.

---

## `finetune.py` — Binary Classifier

### Step 4 — Train MLP classifier (per CV fold)

```python
classifier = Sequential(
    Linear(512 → 512),
    ReLU(),
    Dropout(0.1),
    Linear(512 → 2),     # logit for [healthy, diseased]
)
loss = cross_entropy(logits, labels, weight=class_weights)
```

**Class weights:** `weight[c] = N / (2 × count[c])`. With ~21,444 healthy and ~9,267 diseased in train, the diseased class is weighted ~2.3× higher. This prevents the model from simply predicting healthy for every sample (which would give 70% accuracy but 50% AUROC).

Training runs for 50 epochs per fold with AdamW (lr=1e-4, weight_decay=1e-5) and gradient clipping at 1.0.

### Step 5 — 10-fold study-level GroupKFold CV

```python
gkf = GroupKFold(n_splits=10)
for fold_tr_idx, fold_va_idx in gkf.split(all_idx, labels, study_ids):
    clf = train_classifier(train_emb[fold_tr_idx], ...)
    metrics = eval_classifier(clf, train_emb[fold_va_idx], ...)
```

`GroupKFold` ensures no study appears in both the fold train and fold val. With 222 unique `group_id` values and 10 folds, each fold holds out ~22 studies.

Metrics per fold: `accuracy`, `macro_f1`, `macro_auroc`. Averaged and reported as CV mean ± std.

### Step 6 — External validation

```python
final_clf = train_classifier(train_emb, train_lab, ...)   # all 30,711 train samples
ext_metrics = eval_classifier(final_clf, val_emb, val_lab, ...)
```

The 2,731 val-split samples (14 studies) were never seen during pretraining, fine-tuning, or feature fitting. This is the primary generalization estimate.

### Step 7 — Save results

```
results/finetune_results.json
{
  "cv": {
    "folds":  [ {accuracy, macro_f1, macro_auroc} × 10 ],
    "mean":   {accuracy, macro_f1, macro_auroc},
    "std":    {accuracy, macro_f1, macro_auroc}
  },
  "external": {accuracy, macro_f1, macro_auroc},
  "biomegpt_targets": {
    "cv_accuracy": 0.837,  "cv_macro_auroc": 0.917,
    "external_accuracy": 0.749, "external_macro_auroc": 0.810
  }
}
```

`classifier.pt` is also saved — the final MLP trained on all train embeddings.

---

## `finetune_ovr.py` — Per-Disease OvR Classifier

Runs after `finetune.py` (requires the embedding cache). Controlled by `task.mode` in `config.yaml`; exits if `mode = binary`.

### Step 4 — Discover disease list

```python
diseases = _build_disease_list(meta, min_samples=50, focus=focus_diseases)
```

Reads the raw `label` column. Excludes `"Healthy"` and composite labels (containing `";"`). Keeps diseases with ≥ 50 train samples. In `focused_ovr` mode, further filters to `task.focus_diseases`. Warns about any listed disease that is absent or undersized.

### Step 5 — Build per-disease subsets

For each disease, the full embedding tensor is sliced using a precomputed positional index map:

```python
tr_key_to_idx = {sample_key: row_position_in_train_emb, ...}

# Disease + Healthy rows from train
mask = (train_label == disease) | (train_label == "Healthy")
tr_idx = [tr_key_to_idx[k] for k in dm.index]
tr_emb = train_emb_full[tr_idx]      # [n_disease + n_healthy, 512]
tr_lab = 1 if disease else 0          # OvR binary label
```

This avoids re-running the encoder and avoids re-loading the full dataset per disease.

### Step 6 — Internal StratifiedKFold sanity CV per disease

```python
skf = StratifiedKFold(n_splits=min(n_cv_folds, n_pos, n_neg), shuffle=True)
for fold_tr, fold_va in skf.split(tr_idx, tr_lab):
    clf = _train_clf(tr_emb[fold_tr], tr_lab[fold_tr], ...)
    metrics = _eval_clf(clf, tr_emb[fold_va], tr_lab[fold_va], ...)
```

OvR internal CV is sample-level and should be read only as a sanity/checkpoint metric. Disease positives are concentrated in too few studies for valid 10-fold study-held-out OvR CV; external validation remains the primary generalization metric.

The MLP architecture is identical to `finetune.py`. Class weights are recomputed per disease based on disease sample count vs. healthy sample count — diseases with very few samples (close to the 50-sample minimum) will have a heavily upweighted positive class.

Metrics: `binary_f1` and `auroc` (not macro — binary classification).

### Step 7 — Final classifier + external validation

```python
final_clf = _train_clf(tr_emb, tr_lab, ...)   # all disease + healthy train samples (excl. holdout)
```

External val source depends on the disease:

| Disease           | External val source                                                |
| ----------------- | ------------------------------------------------------------------ |
| Colorectal cancer | val split (202 disease samples)                                    |
| IBD               | val split (204 disease samples)                                    |
| Obesity           | holdout: `gmhi:V-12_Obesity` (104 samples) + val-split Healthy     |
| Type 2 diabetes   | holdout: `cmd:MetaCardis_2020_a` (549 samples) + val-split Healthy |
| Liver Cirrhosis   | weak sample split within the single positive study + val-split Healthy |

For holdout-based diseases: holdout samples were excluded from training. External val = disease samples from the holdout study + all Healthy samples from the existing val split.

For val-split diseases (IBD, Colorectal cancer): existing val split provides both disease and healthy samples.

`external_source` is recorded in `finetune_ovr_results.json` per disease so the origin of each metric is traceable.

### Step 8 — Save results

```
results/finetune_ovr_results.json
{
  "diseases": {
    "CRC": {
      "cv": {
        "folds": [ {binary_f1, auroc, n_pos, n_neg} × 10 ],
        "mean":  {binary_f1, auroc}
      },
      "external":    {binary_f1, auroc},   # or null
      "n_train_pos": 312,
      "n_train_neg": 21444
    },
    "IBD": { ... },
    ...
  },
  "summary": {
    "n_diseases": 5,
    "cv_macro_auroc_mean": ...,
    "cv_macro_auroc_std":  ...,
    "external_macro_auroc_mean": ...,    # averaged over diseases with ext val
    "external_macro_auroc_std":  ...,
    "n_diseases_with_external_val": 2
  }
}
```

---

## Reading the Results

### Binary (`finetune.py`) targets

| Evaluation   | Accuracy | Macro F1 | Macro AUROC |
| ------------ | -------- | -------- | ----------- |
| 10-fold CV   | 83.7%    | 83.0%    | 91.7%       |
| External val | 74.9%    | 73.9%    | 81.0%       |

These are the BiomeGPT paper targets. Match them before drawing any conclusions from Phase 2 experiments.

### OvR (`finetune_ovr.py`) comparison table

```
disease             | CV AUROC (MLP) | CV AUROC (baseline) | Ext AUROC (MLP)        | Ext AUROC (baseline)
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Colorectal cancer   |       ?        |          ?          | ? (val split)          | ? (val split)
IBD                 |       ?        |          ?          | ? (val split)          | ? (val split)
Obesity             |       ?        |          ?          | ? (holdout study)      | ? (holdout study)
Type 2 diabetes     |       ?        |          ?          | ? (holdout study)      | ? (holdout study)
Liver Cirrhosis     |       ?        |          ?          | n/a (1 study only)     | n/a (1 study only)
```

A high CV AUROC that collapses on external val signals study-level overfitting, not generalizable disease signal. A consistent gap between transformer and baseline on external val is the key indicator that pretraining is doing useful work.
