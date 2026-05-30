# Phase 1 — BiomeGPT Baseline Pipeline

Faithful reproduction of [Medearis 2025 BiomeGPT]\(../../sources/Microbiome\ Project/Medearis\ et\ al.\ -\ BiomeGPT\ A\ foundation\ model\ for\ the\ human\ gut\ microbiome.pdf) on the combined cMD + GMHI + GMWI2 dataset.

## Quick Start

```bash
cd experiments/phase1_biomegpt_baseline
python pretrain.py    # ~12 min on MPS/GPU, longer on CPU
python finetune.py    # fast — extracts embeddings once, then trains MLP
python baselines.py   # RF / LR / XGBoost classicals
```

Results land in `results/`. Compare against BiomeGPT targets:

| Evaluation   | Accuracy | Macro F1 | Macro AUROC |
| ------------ | -------- | -------- | ----------- |
| 10-fold CV   | 83.7%    | 83.0%    | 91.7%       |
| External val | 74.9%    | 73.9%    | 81.0%       |

---

## Full Pipeline

```
raw data (CSVs)
      │
      ▼
 dataset.py  ──── load_data()           align abundance matrix + metadata
              ──── compute_bin_edges()  global quantile cut-points (train only)
              ──── MicrobiomeDataset    sparse (species_idx, bin_idx) per sample
              ──── collate_fn           pad + prepend [CLS]
      │
      ├──► pretrain.py  ──► results/pretrain_checkpoint.pt
      │         masked abundance prediction (MSE, 30 epochs)
      │
      ├──► finetune.py  ──► results/finetune_results.json
      │         frozen encoder → [CLS] embeddings → MLP → 10-fold CV + ext val
      │
      └──► baselines.py ──► results/baseline_results.json
                RF / LR / XGBoost on proportions / log1p / CLR / binned
```

---

## Data (`dataset.py`)

### Input files

| File                                              | Shape          | Description                                     |
| ------------------------------------------------- | -------------- | ----------------------------------------------- |
| `relative_abundance_species_proportions_wide.csv` | 33,442 × 2,817 | Row-normalized proportions (sum ≤ 1)            |
| `metadata.csv`                                    | 33,442 rows    | `sample_key`, `split`, `is_healthy`, `group_id` |

`split`: `"train"` (30,711 samples, 222 studies) or `"val"` (2,731 samples, 14 studies).\
`is_healthy`: binary label — `1` = healthy, `0` = any disease.

### `load_data()`

1. Load both CSVs, deduplicate on `sample_key`.
2. Normalize `is_healthy` strings → `int` (handles `"True"/"False"` from CSV).
3. Fill missing `group_id` from `study_id`.
4. Inner-join on `sample_key` → aligned DataFrames.

### Global quantile binning (`compute_bin_edges`)

BiomeGPT maps every nonzero abundance to one of 100 bins. This is done **once**, using only train-split values, so the val split cannot influence the bin boundaries.

```
nonzero values in train:  [0.001, 0.002, ..., 0.95]  (very right-skewed)
                                    ↓
     compute 99 interior percentile cut-points (1%, 2%, ..., 99%)
                                    ↓
     bin_edges: [0.00012, 0.00021, ..., 0.1880]   shape [99]
```

At inference time for any value `v`:

```python
bin_id = searchsorted(bin_edges, v, side='right') + 1   # → 1..100
# absent species → bin_id = 0
```

This gives 101 distinct bin values: `0` (absent) + `1–100` (nonzero abundance).\
Bin `101` is reserved for `[MASK]` during pretraining.

**Why quantile binning?** Microbiome abundances are heavily right-skewed. Quantile binning ensures each bin contains roughly the same number of observations, so no bin is starved of training signal.

### `MicrobiomeDataset`

For each sample, stores only the nonzero species (sparse representation):

```
sample_i  →  species_idx: [142, 307, 891, ...]   (which columns are nonzero)
             bin_ids:     [  5,  23,  78, ...]   (their quantile bins 1-100)
             label:       1 or 0
             study_id:    "cmd:HMP_2012"
```

Average nonzero species per sample: **83** (max 287, out of 2,817 total). Most species are absent in any given sample — the sparse representation is ~34× smaller than a dense row.

### `collate_fn`

Converts a list of variable-length samples into a padded batch:

```
Sample A: 83 species  →  [CLS, s1, s2, ..., s83]             length 84
Sample B: 120 species →  [CLS, s1, s2, ..., s120]            length 121
                                    ↓  padding
species_ids:   [B, 122]   CLS=2817, real species 0..2816, PAD=0 (masked out)
bin_ids:       [B, 122]   bins 0..100, PAD=0 (masked out)
key_padding_mask: [B, 122]  True = ignore this position
```

`[CLS]` is always at position 0. Its species index is `n_species = 2817` (one beyond the real vocabulary).

---

## Model (`model.py`)

### Architecture overview

```
Input per sample: [CLS, sp_1, sp_2, ..., sp_k]   k = nonzero species (avg 83)

For each token at position i:
    species_id  →  Embedding(2818, 512)  →  LayerNorm  →  s   [512]
    bin_id      →  Embedding(102, 512)   →  LayerNorm  →  b   [512]
    token_emb   =  s + b    (for [CLS]: b is zeroed out)

No positional embedding — taxon order is biologically meaningless.

token_emb  →  8 × TransformerEncoderLayer(d=512, heads=8, ffn=512)
           →  hidden states   [B, L, 512]

hidden[:, 0]  →  [CLS] embedding   [B, 512]
```

### Embedding tables

| Table         | Size       | Indices                               |
| ------------- | ---------- | ------------------------------------- |
| `species_emb` | 2818 × 512 | 0..2816 = species, 2817 = [CLS]       |
| `bin_emb`     | 102 × 512  | 0..100 = abundance bins, 101 = [MASK] |

**[CLS] token:** receives only the species embedding (`s`). The bin embedding `b` is set to zero via `masked_fill`. This is because [CLS] has no associated abundance value.

**Why separate LayerNorm for each embedding?** The two embeddings live on different scales. Normalizing each before summing prevents one from drowning the other.

### Transformer encoder

8 stacked `TransformerEncoderLayer` blocks, each with:

- Multi-head self-attention (8 heads, d_model=512, d_head=64)
- Feed-forward sub-network: Linear(512→512) → ReLU → Linear(512→512)
- Layer normalization (post-norm, matching original paper)
- Dropout 0.1

`key_padding_mask` tells the attention to ignore padding positions. No `attn_mask` — every non-padding token can attend to every other.

**Note on masking during pretraining:** The paper says masked organisms are "excluded from attention." In this implementation the masked tokens are still present in the sequence (with `bin_id=101`), so other tokens can see the [MASK] embedding. This is standard BERT style. Excluding them from keys/values would require a per-sample custom attention mask and adds complexity without changing the result much, since there is no positional information that leaks through [MASK].

### Two heads (shared encoder)

```python
# Pretraining head — used only during pretrain.py
pretrain_head = Linear(512 → 1)      # predict bin value (scalar, MSE loss)

# Classifier head — used only during finetune.py
classifier = Linear(512 → 512) → ReLU → Dropout(0.1) → Linear(512 → 2)
```

---

## Pretraining (`pretrain.py`)

Trains the encoder to reconstruct masked species abundances from their neighbors — purely self-supervised, no disease labels used.

### Masking procedure (`apply_mask`)

```
for each sample in batch:
    eligible positions = nonzero species (bin > 0) excluding CLS
    n_mask = max(1, floor(n_eligible × 0.25))
    randomly pick n_mask positions
    replace their bin_id with MASK_BIN_ID (101)
```

### Loss

```python
preds  = pretrain_head(hidden[mask_positions])   # [n_masked]  scalar predictions
targets = original_bin_ids[mask_positions].float()  # [n_masked]  true bins 1..100
loss = MSE(preds, targets)
```

MSE is used rather than cross-entropy because bins are **ordered**: predicting bin 39 when the true bin is 40 should be penalized less than predicting bin 2. MSE encodes this ordinal structure automatically.

### Training schedule

- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Schedule: linear warmup for 1000 steps, then cosine decay to zero
- 30 epochs, batch size 64
- Best checkpoint saved (lowest loss) to `results/pretrain_checkpoint.pt`

The checkpoint contains `model_state`, `species_list`, and `bin_edges` — everything needed to reproduce the exact tokenization for fine-tuning.

---

## Fine-tuning (`finetune.py`)

Trains a binary classifier (healthy vs. diseased) on top of the frozen encoder.

### Why freeze the encoder?

The paper explicitly does this. The encoder learned general microbiome structure during self-supervised pretraining; the classifier learns to read the `[CLS]` embedding for disease signal. Freezing prevents catastrophic forgetting and keeps fine-tuning fast.

### Efficient embedding extraction

Instead of running the full encoder every batch during classifier training (which would recompute the same frozen representations repeatedly), `finetune.py` extracts all `[CLS]` embeddings in **one forward pass** per split:

```
encoder (frozen, eval mode)
    ↓ single pass over all 30,711 train samples
train_emb  [30711, 512]   stored on CPU
val_emb    [2731, 512]    stored on CPU
```

This cache is saved to `results/cls_emb_epoch{N}.pt`. Subsequent finetune runs skip extraction entirely.

**Equivalence:** Since the encoder is frozen and in `eval()` mode (no dropout), running it once and caching is identical to running it every epoch.

### Classifier training

```
Input:  train_emb[fold_train_idx]   [~27640, 512]
            ↓
        Linear(512→512) → ReLU → Dropout(0.1) → Linear(512→2)
            ↓
        cross_entropy(logits, labels, weight=class_weights)
```

**Class weights:** `weight[c] = N / (2 × count[c])`. With ~21,444 healthy and ~9,267 diseased in train, this roughly doubles the loss contribution from diseased samples. This prevents the classifier from ignoring the minority class.

### 10-fold study-level GroupKFold CV

```
train studies (222 unique group_ids)
    ↓  GroupKFold(n_splits=10)
fold 0:  train on 200 studies, eval on 22
fold 1:  train on 200 studies, eval on 22
...
fold 9:  train on 200 studies, eval on 22
```

`GroupKFold` guarantees **no sample from a given study appears in both the fold train and fold val sets**. This reflects real-world deployment: you train on existing cohorts and predict on new ones.

For each fold: fresh MLP classifier, 50 epochs of training, evaluate with accuracy / macro F1 / macro AUROC.

### External validation

After CV, train one final classifier on **all 30,711 train embeddings** and evaluate on the **2,731 val-split samples** (14 studies never seen during pretraining or fine-tuning).

---

## Classical Baselines (`baselines.py`)

Provides the "how much does the transformer actually help?" comparison.

### Feature representations

| Name          | Transform                            | Why                                       |
| ------------- | ------------------------------------ | ----------------------------------------- |
| `proportions` | raw relative abundance               | simplest baseline                         |
| `log1p`       | `log(1 + x)`                         | compresses large values, keeps zeros at 0 |
| `clr`         | `log(x + ε) − mean(log(x + ε))`      | standard compositional data transform     |
| `binned`      | same 100-bin quantile as model input | apples-to-apples with BiomeGPT input      |

CLR uses pseudocount `ε = 1e-6` to handle zeros without losing sparsity structure. The geometric mean in CLR is computed per-sample across all 2,817 species.

### Classifiers

| Classifier              | Key settings                                                                 |
| ----------------------- | ---------------------------------------------------------------------------- |
| **Logistic Regression** | L1 regularization (SAGA solver), `class_weight="balanced"`, `StandardScaler` |
| **Random Forest**       | 300 trees, `max_features="sqrt"`, `class_weight="balanced"`                  |
| **XGBoost**             | 300 trees, `scale_pos_weight=neg/pos` for imbalance, `colsample_bytree=0.5`  |

Same 10-fold study GroupKFold CV + external val as the transformer.

---

## Configuration (`config.yaml`)

All hyperparameters are in one place — no hardcoded values anywhere in the Python files.

```yaml
data:
  n_bins: 100          # number of abundance bins (1..100, 0=absent, 101=MASK)

model:
  d_model: 512         # embedding + attention dimension
  n_layers: 8          # transformer encoder blocks
  n_heads: 8           # attention heads (d_head = 512/8 = 64)
  ffn_dim: 512         # feed-forward hidden dim (same as d_model, per paper)
  dropout: 0.1

pretrain:
  mask_rate: 0.25      # 25% of nonzero species masked per sample
  epochs: 30
  batch_size: 64
  warmup_steps: 1000   # linear warmup before cosine decay

finetune:
  epochs: 50           # classifier training epochs per fold
  n_cv_folds: 10       # study-level GroupKFold
```

---

## What Goes Where

```
results/
  pretrain_checkpoint.pt      encoder weights + species_list + bin_edges
  pretrain_history.json       epoch-by-epoch loss curve
  cls_emb_epoch{N}.pt         cached [CLS] embeddings (reused by finetune)
  classifier.pt               final MLP trained on all train data
  finetune_results.json       CV folds + mean/std + external val metrics
  baseline_results.json       all feature × classifier combinations
```

---

## Design Decisions vs. BiomeGPT Paper

| Aspect                    | Paper                                        | This implementation                              | Impact                                                               |
| ------------------------- | -------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| Binning                   | "100 bins"                                   | Global quantile on train nonzero values          | Faithful; quantile is the natural interpretation for skewed data     |
| Abundance emb             | "MLP with ReLU"                              | `nn.Embedding(102, 512)` lookup                  | Equivalent: lookup table = MLP with identity input; simpler          |
| Masked token handling     | "excluded from attention"                    | Standard BERT [MASK] token (not excluded)        | Minor: no positional encoding means exclusion gives no extra benefit |
| Fine-tuning               | Freeze encoder + MLP                         | Same; embeddings pre-extracted for speed         | Identical results, faster training                                   |
| Class imbalance           | Gaussian noise augmentation                  | Inverse-frequency class weights in cross-entropy | Different mechanism; avoids synthetic-sample artifacts               |
| LassoCV feature selection | Used for disease-specific binary classifiers | Not used for healthy/diseased baseline           | Paper uses it for 33-class problem; baseline is binary               |
| Data                      | cMD + GMHI + GMWI2, 13,524 train             | Same sources, 30,711 train (row-normalized)      | More data due to including GMWI2 fully                               |
