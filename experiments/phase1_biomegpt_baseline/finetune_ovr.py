"""
Phase 1 — per-disease one-vs-rest (OvR) fine-tuning.

For each disease label with >= min_disease_samples in the train split, trains a
binary MLP classifier (disease=1 vs. healthy=0) on frozen [CLS] embeddings.
Runs 10-fold study-level GroupKFold CV and, where the disease appears in the
val split, also reports external-validation metrics.

Usage:
    python finetune_ovr.py

Requires results/cls_emb_epoch<N>.pt (from finetune.py) or the pretrain
checkpoint to re-extract embeddings.  Results saved to
results/finetune_ovr_results.json.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import yaml

from dataset import MicrobiomeDataset, load_data, make_collate_fn
from model import BiomeGPT


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    return {
        "binary_f1": float(f1),
        "auroc": float(auroc),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


@torch.no_grad()
def _extract_embeddings(
    encoder: BiomeGPT,
    dataset: MicrobiomeDataset,
    n_species: int,
    batch_size: int,
    device: torch.device,
    desc: str = "extracting",
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(n_species),
        num_workers=0,
    )
    all_emb, all_lab = [], []
    for species_ids, bin_ids, pad_mask, labels in tqdm(
        loader, desc=f"  {desc}", unit="batch", leave=False
    ):
        emb = encoder.get_cls_embeddings(
            species_ids.to(device), bin_ids.to(device), pad_mask.to(device)
        )
        all_emb.append(emb.cpu())
        all_lab.append(labels)
    return torch.cat(all_emb, 0), torch.cat(all_lab, 0)


def _make_classifier(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 2),
    )


def _train_clf(
    emb_train: torch.Tensor,
    lab_train: torch.Tensor,
    cfg: dict,
    device: torch.device,
    section: str = "finetune_ovr",
) -> nn.Sequential:
    d_model = emb_train.shape[1]
    clf = _make_classifier(d_model, cfg["model"]["dropout"]).to(device)

    counts = torch.bincount(lab_train, minlength=2)
    safe_counts = counts.float().clamp(min=1)
    class_weights = (len(lab_train) / (2.0 * safe_counts)).to(device)

    optimizer = torch.optim.AdamW(
        clf.parameters(),
        lr=cfg[section]["lr"],
        weight_decay=cfg[section]["weight_decay"],
    )
    B = cfg[section]["batch_size"]
    N = len(lab_train)

    for _ in trange(cfg[section]["epochs"], desc="  epochs", leave=False, unit="epoch"):
        perm = torch.randperm(N)
        clf.train()
        for start in range(0, N, B):
            idx = perm[start : start + B]
            emb_b = emb_train[idx].to(device)
            lab_b = lab_train[idx].to(device)
            logits = clf(emb_b)
            loss = F.cross_entropy(logits, lab_b, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), cfg[section]["max_grad_norm"])
            optimizer.step()

    return clf


@torch.no_grad()
def _eval_clf(
    clf: nn.Module, emb: torch.Tensor, lab: torch.Tensor, device: torch.device
) -> dict:
    clf.eval()
    logits = clf(emb.to(device))
    probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return _compute_metrics(lab.numpy(), probs)


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


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    task_cfg = cfg.get("task", {})
    mode = task_cfg.get("mode", "all_ovr")
    focus_diseases: list[str] = task_cfg.get("focus_diseases") or []

    if mode == "binary":
        tqdm.write("mode='binary' — use finetune.py for binary healthy/diseased classification.")
        return

    set_seed(cfg["finetune_ovr"]["seed"])
    device = _get_device()
    tqdm.write(f"Device: {device}  |  mode: {mode}")

    # ── Load pretrained encoder ───────────────────────────────────────────────
    ckpt_path = os.path.join(base_dir, cfg["pretrain"]["checkpoint_path"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {ckpt_path}\nRun pretrain.py first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    species_list: list[str] = ckpt["species_list"]
    bin_edges = np.array(ckpt["bin_edges"])
    n_species = len(species_list)

    encoder = BiomeGPT(
        n_species=n_species,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ffn_dim=cfg["model"]["ffn_dim"],
        dropout=cfg["model"]["dropout"],
        n_bins=cfg["data"]["n_bins"],
    )
    encoder.load_state_dict(ckpt["model_state"])
    encoder = encoder.to(device)
    encoder.eval()
    tqdm.write(f"Loaded encoder (epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f})")

    # ── Load data ─────────────────────────────────────────────────────────────
    abun_path = os.path.join(base_dir, cfg["data"]["abundance_path"])
    meta_path = os.path.join(base_dir, cfg["data"]["metadata_path"])
    abun, meta = load_data(abun_path, meta_path)

    # Re-attach the label column (load_data normalises is_healthy but keeps label)
    raw_meta = pd.read_csv(meta_path, low_memory=False)
    if "sample_key.1" in raw_meta.columns:
        raw_meta = raw_meta.drop(columns=["sample_key.1"])
    raw_meta = raw_meta.set_index("sample_key")
    raw_meta = raw_meta[~raw_meta.index.duplicated(keep="first")]
    # Align with the cleaned meta index
    label_series = raw_meta["label"].reindex(meta.index)
    meta = meta.copy()
    meta["label"] = label_series

    diseases = _build_disease_list(
        meta,
        cfg["finetune_ovr"]["min_disease_samples"],
        focus=focus_diseases if mode == "focused_ovr" else None,
    )
    tqdm.write(f"\nDisease list ({len(diseases)} diseases):")
    for d in diseases:
        n_train = ((meta["split"] == "train") & (meta["label"] == d)).sum()
        tqdm.write(f"  {d}: {n_train} train samples")

    # ── Extract full-corpus [CLS] embeddings (reuse finetune.py cache if present) ──
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    emb_cache = os.path.join(results_dir, f"cls_emb_epoch{ckpt['epoch']}.pt")

    if os.path.exists(emb_cache):
        tqdm.write("\nLoading cached CLS embeddings…")
        cache = torch.load(emb_cache, map_location="cpu", weights_only=False)
        train_emb_full = cache["train_emb"]
        train_lab_full = cache["train_lab"]  # is_healthy labels (not used for OvR)
        val_emb_full = cache["val_emb"]
        # val labels are also is_healthy; we need val label strings separately
    else:
        tqdm.write("\nExtracting CLS embeddings (train)…")
        bs = cfg["finetune_ovr"]["batch_size"] * 4
        train_ds_full = MicrobiomeDataset(
            abun, meta, species_list, bin_edges,
            split="train", n_bins=cfg["data"]["n_bins"],
        )
        train_emb_full, train_lab_full = _extract_embeddings(
            encoder, train_ds_full, n_species, bs, device, desc="train"
        )
        tqdm.write("Extracting CLS embeddings (val)…")
        val_ds_full = MicrobiomeDataset(
            abun, meta, species_list, bin_edges,
            split="val", n_bins=cfg["data"]["n_bins"],
        )
        val_emb_full, _ = _extract_embeddings(
            encoder, val_ds_full, n_species, bs, device, desc="val"
        )
        torch.save(
            {"train_emb": train_emb_full, "train_lab": train_lab_full,
             "val_emb": val_emb_full, "val_lab": _},
            emb_cache,
        )

    # Build index maps: sample_key → row index in the full embedding tensors
    train_meta = meta[meta["split"] == "train"].copy()
    val_meta = meta[meta["split"] == "val"].copy()
    # Ensure alignment: meta rows must match dataset order (same as MicrobiomeDataset)
    train_keys = list(train_meta.index.intersection(abun.index))
    val_keys = list(val_meta.index.intersection(abun.index))
    train_key_to_idx = {k: i for i, k in enumerate(train_keys)}
    val_key_to_idx = {k: i for i, k in enumerate(val_keys)}

    # ── Per-disease OvR loop ──────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    n_cv = cfg["finetune_ovr"]["n_cv_folds"]

    disease_bar = tqdm(diseases, desc="diseases", unit="disease")
    for disease in disease_bar:
        disease_bar.set_postfix(disease=disease[:30])

        # --- Build disease-specific train subset ---
        train_mask = (train_meta["label"] == disease) | (train_meta["label"] == "Healthy")
        dm = train_meta[train_mask].copy()
        dm["_ovr"] = (dm["label"] == disease).astype(int)

        tr_idx = np.array([train_key_to_idx[k] for k in dm.index if k in train_key_to_idx])
        if len(tr_idx) == 0:
            tqdm.write(f"  [{disease}] no train samples found — skip")
            continue
        tr_emb = train_emb_full[tr_idx]
        tr_lab = torch.tensor(dm.loc[[k for k in dm.index if k in train_key_to_idx], "_ovr"].values, dtype=torch.long)
        tr_study = dm.loc[[k for k in dm.index if k in train_key_to_idx], "group_id"].values

        n_pos = int(tr_lab.sum())
        n_neg = int((tr_lab == 0).sum())
        if n_pos == 0 or n_neg == 0:
            tqdm.write(f"  [{disease}] degenerate split (pos={n_pos} neg={n_neg}) — skip")
            continue

        # --- GroupKFold CV ---
        gkf = GroupKFold(n_splits=min(n_cv, len(np.unique(tr_study))))
        cv_results: list[dict] = []
        fold_bar = tqdm(
            enumerate(gkf.split(tr_idx, tr_lab.numpy(), tr_study)),
            total=gkf.get_n_splits(),
            desc=f"  [{disease[:20]}] CV",
            unit="fold",
            leave=False,
        )
        for _, (fold_tr, fold_va) in fold_bar:
            clf = _train_clf(tr_emb[fold_tr], tr_lab[fold_tr], cfg, device)
            metrics = _eval_clf(clf, tr_emb[fold_va], tr_lab[fold_va], device)
            cv_results.append(metrics)
            fold_bar.set_postfix(auroc=f"{metrics['auroc']:.3f}")

        cv_mean = {k: float(np.mean([r[k] for r in cv_results if not np.isnan(r[k])]))
                   for k in cv_results[0] if k not in ("n_pos", "n_neg")}

        # --- Train final classifier → val evaluation ---
        final_clf = _train_clf(tr_emb, tr_lab, cfg, device)

        val_disease_mask = (val_meta["label"] == disease) | (val_meta["label"] == "Healthy")
        vm = val_meta[val_disease_mask].copy()
        vm["_ovr"] = (vm["label"] == disease).astype(int)
        va_idx = np.array([val_key_to_idx[k] for k in vm.index if k in val_key_to_idx])

        external: dict | None = None
        if len(va_idx) > 0:
            va_emb = val_emb_full[va_idx]
            va_lab = torch.tensor(vm.loc[[k for k in vm.index if k in val_key_to_idx], "_ovr"].values, dtype=torch.long)
            if va_lab.sum() > 0 and (va_lab == 0).sum() > 0:
                external = _eval_clf(final_clf, va_emb, va_lab, device)

        tqdm.write(
            f"  {disease[:35]:<35} | "
            f"CV auroc={cv_mean.get('auroc', float('nan')):.4f}  "
            f"f1={cv_mean.get('binary_f1', float('nan')):.4f}  "
            + (f"| ext auroc={external['auroc']:.4f}" if external else "| no ext val")
        )

        all_results[disease] = {
            "cv": {"folds": cv_results, "mean": cv_mean},
            "external": external,
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
        }

    # ── Aggregate summary ─────────────────────────────────────────────────────
    cv_aurocs = [v["cv"]["mean"]["auroc"] for v in all_results.values() if not np.isnan(v["cv"]["mean"]["auroc"])]
    ext_aurocs = [v["external"]["auroc"] for v in all_results.values()
                  if v["external"] is not None and not np.isnan(v["external"]["auroc"])]

    summary = {
        "n_diseases": len(all_results),
        "cv_macro_auroc_mean": float(np.mean(cv_aurocs)) if cv_aurocs else float("nan"),
        "cv_macro_auroc_std": float(np.std(cv_aurocs)) if cv_aurocs else float("nan"),
        "external_macro_auroc_mean": float(np.mean(ext_aurocs)) if ext_aurocs else float("nan"),
        "external_macro_auroc_std": float(np.std(ext_aurocs)) if ext_aurocs else float("nan"),
        "n_diseases_with_external_val": len(ext_aurocs),
    }
    tqdm.write(f"\nOvR summary ({len(all_results)} diseases):")
    tqdm.write(f"  CV macro AUROC:  {summary['cv_macro_auroc_mean']:.4f} ± {summary['cv_macro_auroc_std']:.4f}")
    if ext_aurocs:
        tqdm.write(f"  Ext macro AUROC: {summary['external_macro_auroc_mean']:.4f} ± {summary['external_macro_auroc_std']:.4f} ({len(ext_aurocs)} diseases)")

    results_path = os.path.join(base_dir, cfg["finetune_ovr"]["results_path"])
    with open(results_path, "w") as f:
        json.dump({"diseases": all_results, "summary": summary}, f, indent=2)
    tqdm.write(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
