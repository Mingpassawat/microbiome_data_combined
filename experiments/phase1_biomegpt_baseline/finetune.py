"""
Phase 1 — BiomeGPT fine-tuning: frozen encoder + MLP classifier.

Usage:
    python finetune.py

Loads the pretrained encoder from results/pretrain_checkpoint.pt, freezes it,
pre-extracts [CLS] embeddings for all samples (single pass), then:
  1. Runs 10-fold study-level GroupKFold CV on train embeddings.
  2. Trains a final classifier on all train embeddings and evaluates on val split.

Results saved to results/finetune_results.json.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_auroc": float(auroc),
    }


@torch.no_grad()
def extract_cls_embeddings(
    encoder: BiomeGPT,
    dataset: MicrobiomeDataset,
    n_species: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single forward pass through the frozen encoder to get [CLS] embeddings.
    Returns (embeddings [N, D], labels [N]).
    """
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(n_species),
        num_workers=0,
    )
    all_emb, all_lab = [], []
    for species_ids, bin_ids, pad_mask, labels in tqdm(loader, desc="  extracting", unit="batch", leave=False):
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


def train_classifier(
    emb_train: torch.Tensor,
    lab_train: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> nn.Sequential:
    """Train an MLP classifier on pre-extracted [CLS] embeddings."""
    d_model = emb_train.shape[1]
    clf = _make_classifier(d_model, cfg["model"]["dropout"]).to(device)

    # Class weights for mild label imbalance (healthy vs. diseased)
    counts = torch.bincount(lab_train)
    class_weights = (len(lab_train) / (2.0 * counts.float())).to(device)

    optimizer = torch.optim.AdamW(
        clf.parameters(),
        lr=cfg["finetune"]["lr"],
        weight_decay=cfg["finetune"]["weight_decay"],
    )

    B = cfg["finetune"]["batch_size"]
    N = len(lab_train)

    for _ in trange(cfg["finetune"]["epochs"], desc="  clf epochs", leave=False, unit="epoch"):
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
            torch.nn.utils.clip_grad_norm_(
                clf.parameters(), cfg["finetune"]["max_grad_norm"]
            )
            optimizer.step()

    return clf


@torch.no_grad()
def eval_classifier(
    clf: nn.Module,
    emb: torch.Tensor,
    lab: torch.Tensor,
    device: torch.device,
) -> dict:
    clf.eval()
    logits = clf(emb.to(device))
    probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return _compute_metrics(lab.numpy(), probs)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["finetune"]["seed"])
    device = _get_device()
    print(f"Device: {device}")

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
    print(f"Loaded encoder (pretrain epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f})")

    # ── Load datasets ─────────────────────────────────────────────────────────
    abun_path = os.path.join(base_dir, cfg["data"]["abundance_path"])
    meta_path = os.path.join(base_dir, cfg["data"]["metadata_path"])
    abun, meta = load_data(abun_path, meta_path)

    train_ds = MicrobiomeDataset(
        abun, meta, species_list, bin_edges,
        split="train", n_bins=cfg["data"]["n_bins"],
    )
    val_ds = MicrobiomeDataset(
        abun, meta, species_list, bin_edges,
        split="val", n_bins=cfg["data"]["n_bins"],
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Extract [CLS] embeddings once (frozen encoder) ────────────────────────
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    emb_cache = os.path.join(results_dir, f"cls_emb_epoch{ckpt['epoch']}.pt")

    if os.path.exists(emb_cache):
        print("Loading cached CLS embeddings…")
        cache = torch.load(emb_cache, map_location="cpu", weights_only=False)
        train_emb, train_lab = cache["train_emb"], cache["train_lab"]
        val_emb, val_lab = cache["val_emb"], cache["val_lab"]
    else:
        print("Extracting CLS embeddings (train)…")
        bs = cfg["finetune"]["batch_size"] * 4
        train_emb, train_lab = extract_cls_embeddings(
            encoder, train_ds, n_species, bs, device
        )
        print("Extracting CLS embeddings (val)…")
        val_emb, val_lab = extract_cls_embeddings(
            encoder, val_ds, n_species, bs, device
        )
        torch.save(
            {"train_emb": train_emb, "train_lab": train_lab,
             "val_emb": val_emb, "val_lab": val_lab},
            emb_cache,
        )
    print(f"  train_emb: {train_emb.shape}, val_emb: {val_emb.shape}")

    # ── 10-fold study-level GroupKFold CV ─────────────────────────────────────
    study_ids = np.array(train_ds.study_ids)
    all_idx = np.arange(len(train_ds))
    gkf = GroupKFold(n_splits=cfg["finetune"]["n_cv_folds"])
    cv_results: list[dict] = []

    n_folds = cfg["finetune"]["n_cv_folds"]
    print(f"\nRunning {n_folds}-fold study-level CV…")
    fold_bar = tqdm(
        enumerate(gkf.split(all_idx, train_lab.numpy(), study_ids)),
        total=n_folds, desc="CV folds", unit="fold",
    )
    for fold, (tr_idx, va_idx) in fold_bar:
        clf = train_classifier(
            train_emb[tr_idx], train_lab[tr_idx], cfg, device
        )
        metrics = eval_classifier(clf, train_emb[va_idx], train_lab[va_idx], device)
        cv_results.append(metrics)
        fold_bar.set_postfix(
            auroc=f"{metrics['macro_auroc']:.4f}",
            acc=f"{metrics['accuracy']:.4f}",
        )
        tqdm.write(
            f"  Fold {fold + 1:2d}/{n_folds} | "
            f"acc={metrics['accuracy']:.4f}  f1={metrics['macro_f1']:.4f}  "
            f"auroc={metrics['macro_auroc']:.4f}"
        )

    cv_mean = {k: float(np.mean([r[k] for r in cv_results])) for k in cv_results[0]}
    cv_std = {k: float(np.std([r[k] for r in cv_results])) for k in cv_results[0]}
    print("\n10-fold CV (mean ± std):")
    for k in cv_mean:
        print(f"  {k}: {cv_mean[k]:.4f} ± {cv_std[k]:.4f}")

    # ── Train on all train data → evaluate on external val ───────────────────
    print("\nTraining final classifier (all train)…")
    final_clf = train_classifier(train_emb, train_lab, cfg, device)
    ext_metrics = eval_classifier(final_clf, val_emb, val_lab, device)
    print("External validation:")
    for k, v in ext_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final classifier
    torch.save(
        {"classifier_state": final_clf.state_dict(), "config": cfg},
        os.path.join(results_dir, "classifier.pt"),
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "cv": {"folds": cv_results, "mean": cv_mean, "std": cv_std},
        "external": ext_metrics,
        "biomegpt_targets": {
            "cv_accuracy": 0.837,
            "cv_macro_f1": 0.830,
            "cv_macro_auroc": 0.917,
            "external_accuracy": 0.749,
            "external_macro_f1": 0.739,
            "external_macro_auroc": 0.810,
        },
    }
    results_path = os.path.join(base_dir, cfg["finetune"]["results_path"])
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
