# Pretraining — Step-by-Step

`pretrain.py` trains the BiomeGPT transformer encoder with **BERT-style masked abundance reconstruction**, the self-supervised objective from the BiomeGPT thesis. No labels are used — the model learns microbiome structure by predicting the abundance of randomly hidden species from the rest of the sample.

| Item      | Value                             |
| --------- | --------------------------------- |
| Script    | `pretrain.py`                     |
| Input     | `train`-split samples only        |
| Objective | Masked abundance regression (MSE) |
| Output    | `results/pretrain_checkpoint.pt`  |
| History   | `results/pretrain_history.json`   |

Run it first — every downstream script (`finetune.py`, `finetune_ovr.py`) loads the checkpoint this produces.

```bash
python pretrain.py
```

---

## Background — how a sample is represented

Each sample is a **sparse sequence**: only the species with nonzero abundance are tokenized (see `dataset.py`). A token is a `(species_id, bin_id)` pair.

- **`species_id`** — `0 .. n_species-1` index a taxon; `n_species` is the special `[CLS]` token.
- **`bin_id`** — abundance discretized into bins. `0` = absent (never appears as a real token, only via padding), `1 .. 100` = quantile bins of nonzero proportions, `101` = `[MASK]` (`MASK_BIN_ID`).

The bin edges are the `n_bins-1 = 99` quantile cut-points computed **from train-split nonzero values only** (`compute_bin_edges`). Each batch is prepended with `[CLS]` and right-padded; `key_padding_mask=True` marks padding to be ignored by attention (`collate_fn`).

No positional embeddings are used — taxon order carries no biological meaning.

---

## Step 1 — Load config & data

```python
cfg = yaml.safe_load(open("config.yaml"))
abun, meta = load_data(abundance_path, metadata_path)   # aligned on sample_key
bin_edges  = compute_bin_edges(abun, meta, n_bins=100)  # train-only quantiles
dataset    = MicrobiomeDataset(abun, meta, species_list, bin_edges, split="train")
```

Only `split == "train"` rows enter the dataset. Val-split studies are never seen during pretraining — this keeps later LOSO/val evaluation honest. The bin edges are likewise derived only from train data, then reused everywhere.

The `DataLoader` uses `make_collate_fn(n_species)` to build padded batches and `num_workers=0` (worker forking is unreliable on macOS).

## Step 2 — Build the model

```python
model = BiomeGPT(n_species, d_model=512, n_heads=8, n_layers=8,
                 ffn_dim=512, dropout=0.1, n_bins=100)
```

The encoder is a standard `nn.TransformerEncoder` (8 layers, post-norm). Token embedding is `LayerNorm(species_emb) + LayerNorm(bin_emb)`, with the bin term zeroed for `[CLS]`. For pretraining, only the `pretrain_head` (a single `Linear(d_model, 1)`) is used — the classifier head stays untouched.

## Step 3 — Optimizer & schedule

```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = LambdaLR(optimizer, lr_lambda)   # linear warmup → cosine decay
```

`lr_lambda` ramps linearly over `warmup_steps` (1000), then follows a cosine decay to zero across the remaining `epochs × len(loader)` steps.

---

## Step 4 — The masking objective (the core of pretraining)

This is what makes it self-supervised. For each sample, `apply_mask` selects a fraction (`mask_rate = 0.25`) of its **nonzero, non-`[CLS]`** species and replaces their `bin_id` with `MASK_BIN_ID (101)`. The original bin values become the regression targets.

```python
def apply_mask(bin_ids, key_padding_mask, mask_rate):
    for i in range(B):
        eligible = (~key_padding_mask[i]) & (bin_ids[i] > 0)  # real, present taxa
        eligible[0] = False                                   # never mask [CLS]
        pos = eligible.nonzero().squeeze(-1)
        n_mask = max(1, int(pos.numel() * mask_rate))         # ≥1 mask per sample
        chosen = pos[torch.randperm(pos.numel())[:n_mask]]
        masked[i, chosen] = MASK_BIN_ID                       # hide the abundance
        mask_pos[i, chosen] = True
    return masked, bin_ids.float(), mask_pos
```

Key points:

- Masking is **resampled fresh every batch** — the same sample gets different hidden species across epochs, which is what gives the model varied training signal.
- The **species identity stays visible**; only its abundance bin is hidden. The model must infer _how much_ of a known taxon is present from the surrounding community.
- `[CLS]`, padding, and absent species are never masked.

## Step 5 — Forward, loss, update

```python
preds = model.pretrain_forward(species_ids, masked_bins, pad_mask)  # [B, L] scalars
loss  = F.mse_loss(preds[mask_pos], targets[mask_pos])              # masked positions only
```

The head predicts a **continuous scalar** per position; loss is MSE against the original (integer) bin index, but **only at masked positions** — unmasked tokens contribute no gradient. Then the usual:

```python
loss.backward()
clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
optimizer.step(); scheduler.step()
```

Gradient clipping at norm 1.0 stabilizes early training.

> **Note** — the objective is framed as _regression on bin indices_ (MSE), not classification over 100 bins. This treats neighboring bins as numerically close, so predicting bin 51 when the answer is 50 is nearly free, unlike a cross-entropy over bin classes.

---

## Step 6 — Checkpointing

Best-by-training-loss is saved each time the epoch loss improves:

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "loss": loss,
    "config": cfg,
    "species_list": species_list,   # the 2,817-species vocabulary
    "bin_edges": bin_edges.tolist(), # the 99 quantile cut-points
}, "results/pretrain_checkpoint.pt")
```

The checkpoint **bundles the vocabulary and bin edges** with the weights so that fine-tuning tokenizes inputs exactly as the encoder was trained — different bin edges would silently change the meaning of every token. Per-epoch losses are also dumped to `results/pretrain_history.json`.

There is no held-out validation loss here; "best" is the lowest training reconstruction loss. The real quality check happens downstream, when the frozen encoder's `[CLS]` embeddings are evaluated on the classification tasks.

---

## Key hyperparameters (`config.yaml → pretrain`)

| Param           | Value  | Meaning                                       |
| --------------- | ------ | --------------------------------------------- |
| `mask_rate`     | `0.25` | fraction of nonzero species masked per sample |
| `lr`            | `1e-4` | AdamW peak learning rate                      |
| `weight_decay`  | `1e-5` | AdamW weight decay                            |
| `epochs`        | `30`   | passes over the train split                   |
| `batch_size`    | `64`   | samples per batch                             |
| `warmup_steps`  | `1000` | linear warmup before cosine decay             |
| `max_grad_norm` | `1.0`  | gradient clipping threshold                   |
| `seed`          | `42`   | seeds `random`, `numpy`, `torch`              |

Model size is set under `config.yaml → model` (`d_model`, `n_layers`, `n_heads`, `ffn_dim`, `dropout`) and `data.n_bins`.

---

## Output → next step

`results/pretrain_checkpoint.pt` is consumed by **fine-tuning** — see [`FINETUNE.md`](FINETUNE.md). There the encoder is frozen and only a small MLP head is trained on the `[CLS]` embeddings.
