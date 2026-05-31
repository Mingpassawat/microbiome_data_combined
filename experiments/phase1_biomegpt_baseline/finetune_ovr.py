"""
Phase 1 — per-disease one-vs-rest (OvR) fine-tuning.

Trains one binary MLP classifier per OvR task (disease=1 vs. healthy=0) on
frozen [CLS] embeddings.  Tasks and train/val splits are precomputed by
make_task_splits.py and stored in data/combined_microbiome/task_splits.json.

Runs sample-level shuffled StratifiedKFold CV as an internal sanity metric and
external validation using the precomputed val_pos_keys / val_neg_keys per task.

Usage:
    python make_task_splits.py   # once, to generate task_splits.json
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
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import yaml

from dataset import MicrobiomeDataset, load_data, make_collate_fn
from model import BiomeGPT
from ovr_cv import (
    EXTERNAL_ROLE,
    external_validation_strength,
    finite_values,
    make_ovr_folds,
    summarize_metric_dicts,
)


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
    return {
        "binary_f1": float(f1),
        "auroc": float(auroc),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
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


def _gather_emb(
    keys: list[str],
    train_key_to_idx: dict[str, int],
    val_key_to_idx: dict[str, int],
    train_emb: torch.Tensor,
    val_emb: torch.Tensor,
) -> torch.Tensor:
    """Gather embeddings for a list of keys from whichever cache they belong to."""
    tr_pos = [train_key_to_idx[k] for k in keys if k in train_key_to_idx]
    vl_pos = [val_key_to_idx[k] for k in keys if k in val_key_to_idx]
    parts: list[torch.Tensor] = []
    if tr_pos:
        parts.append(train_emb[tr_pos])
    if vl_pos:
        parts.append(val_emb[vl_pos])
    if not parts:
        return torch.empty(0, train_emb.shape[1])
    return torch.cat(parts, 0)


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
    class_weights = (len(lab_train) / (2.0 * counts.float().clamp(min=1))).to(device)

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
            logits = clf(emb_train[idx].to(device))
            loss = F.cross_entropy(logits, lab_train[idx].to(device), weight=class_weights)
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
    probs = F.softmax(clf(emb.to(device)), dim=-1)[:, 1].cpu().numpy()
    return _compute_metrics(lab.numpy(), probs)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("task", {}).get("mode", "ovr")
    if mode == "binary":
        tqdm.write("mode='binary' — use finetune.py for binary healthy/diseased classification.")
        return

    set_seed(cfg["finetune_ovr"]["seed"])
    device = _get_device()
    tqdm.write(f"Device: {device}")

    # ── Load task splits ──────────────────────────────────────────────────────
    splits_path = os.path.join(base_dir, cfg["data"]["task_splits_path"])
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"task_splits.json not found: {splits_path}\nRun make_task_splits.py first."
        )
    with open(splits_path) as f:
        task_splits: dict[str, dict] = json.load(f)
    ovr_tasks = {k: v for k, v in task_splits.items() if k != "binary"}
    tqdm.write(f"OvR tasks: {list(ovr_tasks)}")

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

    # Attach raw label column (load_data keeps is_healthy but not label string)
    raw_meta = pd.read_csv(meta_path, low_memory=False)
    if "sample_key.1" in raw_meta.columns:
        raw_meta = raw_meta.drop(columns=["sample_key.1"])
    raw_meta = raw_meta.set_index("sample_key")
    raw_meta = raw_meta[~raw_meta.index.duplicated(keep="first")]
    meta = meta.copy()
    meta["label"] = raw_meta["label"].reindex(meta.index)

    # ── Extract [CLS] embeddings (reuse finetune.py cache if present) ─────────
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    emb_cache = os.path.join(results_dir, f"cls_emb_epoch{ckpt['epoch']}.pt")

    if os.path.exists(emb_cache):
        tqdm.write("\nLoading cached CLS embeddings…")
        cache = torch.load(emb_cache, map_location="cpu", weights_only=False)
        train_emb_full = cache["train_emb"]
        train_lab_full = cache["train_lab"]
        val_emb_full   = cache["val_emb"]
    else:
        tqdm.write("\nExtracting CLS embeddings…")
        bs = cfg["finetune_ovr"]["batch_size"] * 4
        train_ds = MicrobiomeDataset(
            abun, meta, species_list, bin_edges,
            split="train", n_bins=cfg["data"]["n_bins"],
        )
        train_emb_full, train_lab_full = _extract_embeddings(
            encoder, train_ds, n_species, bs, device, desc="train"
        )
        val_ds = MicrobiomeDataset(
            abun, meta, species_list, bin_edges,
            split="val", n_bins=cfg["data"]["n_bins"],
        )
        val_emb_full, val_lab_full = _extract_embeddings(
            encoder, val_ds, n_species, bs, device, desc="val"
        )
        torch.save(
            {"train_emb": train_emb_full, "train_lab": train_lab_full,
             "val_emb": val_emb_full, "val_lab": val_lab_full},
            emb_cache,
        )

    # key → row index in the full embedding tensors (same order as MicrobiomeDataset)
    train_meta = meta[meta["split"] == "train"]
    val_meta   = meta[meta["split"] == "val"]
    train_keys_ordered = list(train_meta.index.intersection(abun.index))
    val_keys_ordered   = list(val_meta.index.intersection(abun.index))
    train_key_to_idx = {k: i for i, k in enumerate(train_keys_ordered)}
    val_key_to_idx   = {k: i for i, k in enumerate(val_keys_ordered)}

    # ── Per-disease OvR loop ──────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    n_cv = cfg["finetune_ovr"]["n_cv_folds"]

    disease_bar = tqdm(ovr_tasks.items(), desc="diseases", unit="disease", total=len(ovr_tasks))
    for disease, task in disease_bar:
        disease_bar.set_postfix(disease=disease[:30])

        # ── Train set from precomputed keys ───────────────────────────────────
        tr_keys = [k for k in task["train_keys"] if k in train_key_to_idx]
        if not tr_keys:
            tqdm.write(f"  [{disease}] no train keys in embedding index — skip")
            continue
        tr_pos = np.array([train_key_to_idx[k] for k in tr_keys])
        tr_emb = train_emb_full[tr_pos]
        tr_lab = torch.tensor(
            [1 if meta.loc[k, "label"] == disease else 0 for k in tr_keys],
            dtype=torch.long,
        )

        n_pos = int(tr_lab.sum())
        n_neg = int((tr_lab == 0).sum())
        if n_pos == 0 or n_neg == 0:
            tqdm.write(f"  [{disease}] degenerate split (pos={n_pos} neg={n_neg}) — skip")
            continue

        # ── Internal sample-level sanity CV ───────────────────────────────────
        folds, cv_meta = make_ovr_folds(tr_lab.numpy(), n_cv, cfg["finetune_ovr"]["seed"])
        cv_results: list[dict] = []
        if cv_meta["cv_status"] == "available":
            fold_bar = tqdm(
                enumerate(folds),
                total=cv_meta["actual_folds"],
                desc=f"  [{disease[:20]}] CV",
                unit="fold",
                leave=False,
            )
            for _, (fold_tr, fold_va) in fold_bar:
                clf = _train_clf(tr_emb[fold_tr], tr_lab[fold_tr], cfg, device)
                metrics = _eval_clf(clf, tr_emb[fold_va], tr_lab[fold_va], device)
                cv_results.append(metrics)
                fold_bar.set_postfix(auroc=f"{metrics['auroc']:.3f}")
        else:
            tqdm.write(f"  [{disease}] internal CV unavailable: {cv_meta['cv_unavailable_reason']}")

        cv_summary = summarize_metric_dicts(cv_results)
        cv_summary.update(cv_meta)
        cv_mean = cv_summary["mean"]

        # ── Final classifier + external val ───────────────────────────────────
        final_clf = _train_clf(tr_emb, tr_lab, cfg, device)

        pos_emb = _gather_emb(task["val_pos_keys"], train_key_to_idx, val_key_to_idx,
                               train_emb_full, val_emb_full)
        neg_emb = _gather_emb(task["val_neg_keys"], train_key_to_idx, val_key_to_idx,
                               train_emb_full, val_emb_full)

        external: dict | None = None
        ext_source = task.get("val_source", "none")
        ext_strength = external_validation_strength(task)
        if len(pos_emb) > 0 and len(neg_emb) > 0:
            ext_emb = torch.cat([pos_emb, neg_emb], 0)
            ext_lab = torch.cat([
                torch.ones(len(pos_emb), dtype=torch.long),
                torch.zeros(len(neg_emb), dtype=torch.long),
            ], 0)
            external = _eval_clf(final_clf, ext_emb, ext_lab, device)

        tqdm.write(
            f"  {disease[:35]:<35} | "
            f"internal CV auroc={cv_mean.get('auroc', float('nan')):.4f}  "
            f"f1={cv_mean.get('binary_f1', float('nan')):.4f}  "
            + (f"| ext auroc={external['auroc']:.4f} ({ext_source}; {ext_strength})"
               if external else f"| no ext val ({ext_source})")
        )

        all_results[disease] = {
            "cv": cv_summary,
            "external": external,
            "external_role": EXTERNAL_ROLE,
            "external_source": ext_source,
            "external_validation_strength": ext_strength,
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
        }

    # ── Aggregate summary ─────────────────────────────────────────────────────
    cv_aurocs = finite_values(
        v["cv"]["mean"].get("auroc", float("nan")) for v in all_results.values()
    )
    ext_aurocs = finite_values(
        v["external"]["auroc"]
        for v in all_results.values()
        if v["external"] is not None
    )

    summary = {
        "n_diseases": len(all_results),
        "cv_macro_auroc_mean":      float(np.mean(cv_aurocs))  if cv_aurocs  else float("nan"),
        "cv_macro_auroc_std":       float(np.std(cv_aurocs))   if cv_aurocs  else float("nan"),
        "external_macro_auroc_mean": float(np.mean(ext_aurocs)) if ext_aurocs else float("nan"),
        "external_macro_auroc_std":  float(np.std(ext_aurocs))  if ext_aurocs else float("nan"),
        "n_diseases_with_external_val": len(ext_aurocs),
    }
    tqdm.write(f"\nOvR summary ({len(all_results)} diseases):")
    tqdm.write(
        f"  Internal CV macro AUROC:  {summary['cv_macro_auroc_mean']:.4f} "
        f"± {summary['cv_macro_auroc_std']:.4f}"
    )
    if ext_aurocs:
        tqdm.write(
            f"  Ext macro AUROC: {summary['external_macro_auroc_mean']:.4f} "
            f"± {summary['external_macro_auroc_std']:.4f} ({len(ext_aurocs)} diseases)"
        )

    results_path = os.path.join(base_dir, cfg["finetune_ovr"]["results_path"])
    with open(results_path, "w") as f:
        json.dump({"diseases": all_results, "summary": summary}, f, indent=2)
    tqdm.write(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
