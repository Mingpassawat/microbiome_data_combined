# Microbiome Transformer Project

Transformer-based models for predicting human disease from gut microbiome metagenomic profiles.

## Read First

1. `/Users/ming/Documents/Vault/t/AGENTS.md` — vault operating rules
2. `/Users/ming/Documents/Vault/t/wiki/indexes/Microbiome Project.md` — topic index and tracker
3. `/Users/ming/Documents/Vault/t/wiki/notes/Microbiome Project/Microbiome Improvement Plan.md` — phased experiment plan

For domain knowledge look up in the vault before writing code:

- Concepts / metrics / biology: `wiki/pages/concepts/`
- Model families / paradigms: `wiki/pages/paradigms/`
- Architectures / transforms: `wiki/pages/methods/`
- Datasets / tools: `wiki/pages/entities/`
- Papers: `wiki/pages/works/` — especially `Medearis 2025 BiomeGPT.md`
- Syntheses: `wiki/pages/syntheses/` — especially `Improving Transformer-Based Microbiome Explanation.md`

## Data

Default input: `data/combined_microbiome/relative_abundance_species_proportions_wide.csv` + `metadata.csv` (33,442 samples, 2,817 species, sources: cMD + GMHI + GMWI2).

Key metadata columns: `label`, `is_healthy`, `study_id`, `group_id`, `split` (`train`/`val`).

Never use `val`-split studies during pretraining, fine-tuning, model selection, or bin calibration.

## Experiment Plan

**Phase 1 — BiomeGPT baseline**: reproduce species embedding + 100-bin abundance embedding + `[CLS]` + masked abundance pretraining + frozen-encoder classifier. Target: ~83.7% CV accuracy / 91.7% macro AUROC internal, ~74.9% / 81.0% external. Add RF / logistic regression / XGBoost baselines at same split level.

**Phase 2 — Data utilization**: test strict split-first vs. standard BiomeGPT vs. multi-task pretraining vs. frozen embeddings. Key question: does stricter study isolation lower internal CV but make external results more trustworthy?

**Phase 3 — Improvements** (in order): continuous abundance encoding → supervised contrastive learning → abundance-aware set pooling → latent-space domain alignment. First experiment: swap binned abundance embedding for a continuous value projection; keep everything else fixed; compare binned vs. raw proportion vs. log1p vs. CLR on both internal CV and LOSO.

## Experiment Layout

```
experiments/<phase_name>/
  config.yaml       all hyperparameters and paths, nothing hardcoded
  train.py
  evaluate.py
  results/
```

## Rules

- **Training verbosity**: all training scripts must use `tqdm` progress bars. Use `trange` for epoch loops (show live `loss=` / `best=` postfix), `tqdm(loader, ...)` for batch loops (show running loss postfix, `leave=False`), and `tqdm` for CV fold loops (show live `auroc=` postfix). Use `tqdm.write(...)` instead of `print(...)` inside any tqdm loop to avoid interleaved output.
- **Primary metric**: macro AUROC and F1 on study-held-out (LOSO) evaluation — not random-split CV alone.
- Always compare against the reproduced BiomeGPT baseline at the same split level.
- Use `karpathy-guidelines` skill before writing or reviewing code.
- No Jupyter notebooks for experiment code (EDA only).
- After completing work, append a dated bullet to the tracker in `/Users/ming/Documents/Vault/t/wiki/indexes/Microbiome Project.md`.
- When working with library or API (e.g., pytorch, OpenAI API), use `get-api-doc` skill to fetch the lastest document. Your knowledge is often outdated.
