"""
Phase 1 — BiomeGPT pretraining: masked abundance prediction.

Usage:
    python pretrain.py

Trains the transformer encoder on train-split samples using BERT-style masked
abundance reconstruction (25% masking rate, MSE loss on bin indices).
Saves the best checkpoint to results/pretrain_checkpoint.pt.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import yaml

from dataset import MASK_BIN_ID, MicrobiomeDataset, compute_bin_edges, load_data, make_collate_fn
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


def apply_mask(
    bin_ids: torch.Tensor,
    key_padding_mask: torch.Tensor,
    mask_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly mask `mask_rate` fraction of nonzero, non-CLS species per sample.

    Returns:
        masked_bin_ids  bin_ids with chosen positions replaced by MASK_BIN_ID
        targets         original bin_ids as float (regression targets for MSE)
        mask_pos        bool tensor marking masked positions
    """
    B, L = bin_ids.shape
    masked = bin_ids.clone()
    mask_pos = torch.zeros(B, L, dtype=torch.bool, device=bin_ids.device)

    for i in range(B):
        # Eligible: not padding, not CLS (position 0), nonzero abundance (bin > 0)
        eligible = (~key_padding_mask[i]) & (bin_ids[i] > 0)
        eligible[0] = False  # never mask CLS
        pos = eligible.nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() == 0:
            continue
        n_mask = max(1, int(pos.numel() * mask_rate))
        chosen = pos[torch.randperm(pos.numel(), device=pos.device)[:n_mask]]
        masked[i, chosen] = MASK_BIN_ID
        mask_pos[i, chosen] = True

    return masked, bin_ids.float(), mask_pos


def train_one_epoch(
    model: BiomeGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    mask_rate: float,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  batches", leave=False, unit="batch")
    for species_ids, bin_ids, pad_mask, _ in pbar:
        species_ids = species_ids.to(device)
        bin_ids = bin_ids.to(device)
        pad_mask = pad_mask.to(device)

        masked_bins, targets, mask_pos = apply_mask(bin_ids, pad_mask, mask_rate)
        if not mask_pos.any():
            continue

        preds = model.pretrain_forward(species_ids, masked_bins, pad_mask)  # [B, L]
        loss = F.mse_loss(preds[mask_pos], targets[mask_pos])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["pretrain"]["seed"])
    device = _get_device()
    print(f"Device: {device}")

    abun_path = os.path.join(base_dir, cfg["data"]["abundance_path"])
    meta_path = os.path.join(base_dir, cfg["data"]["metadata_path"])
    print("Loading data…")
    abun, meta = load_data(abun_path, meta_path)
    species_list = list(abun.columns)
    n_species = len(species_list)
    print(f"  {len(abun)} total samples, {n_species} species")

    bin_edges = compute_bin_edges(abun, meta, cfg["data"]["n_bins"])

    dataset = MicrobiomeDataset(
        abun, meta, species_list, bin_edges,
        split="train", n_bins=cfg["data"]["n_bins"],
    )
    print(f"  {len(dataset)} train samples")

    loader = DataLoader(
        dataset,
        batch_size=cfg["pretrain"]["batch_size"],
        shuffle=True,
        collate_fn=make_collate_fn(n_species),
        num_workers=0,  # >0 can cause issues on macOS with fork
    )

    model = BiomeGPT(
        n_species=n_species,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ffn_dim=cfg["model"]["ffn_dim"],
        dropout=cfg["model"]["dropout"],
        n_bins=cfg["data"]["n_bins"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["pretrain"]["lr"],
        weight_decay=cfg["pretrain"]["weight_decay"],
    )

    total_steps = cfg["pretrain"]["epochs"] * len(loader)
    warmup = cfg["pretrain"]["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(base_dir, cfg["pretrain"]["checkpoint_path"])

    history: list[dict] = []
    best_loss = float("inf")

    epoch_bar = trange(1, cfg["pretrain"]["epochs"] + 1, desc="Pretraining", unit="epoch")
    for epoch in epoch_bar:
        loss = train_one_epoch(
            model, loader, optimizer, scheduler, device,
            mask_rate=cfg["pretrain"]["mask_rate"],
            max_grad_norm=cfg["pretrain"]["max_grad_norm"],
        )
        history.append({"epoch": epoch, "loss": loss})
        epoch_bar.set_postfix(loss=f"{loss:.4f}", best=f"{min(best_loss, loss):.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss,
                    "config": cfg,
                    "species_list": species_list,
                    "bin_edges": bin_edges.tolist(),
                },
                ckpt_path,
            )

    with open(os.path.join(results_dir, "pretrain_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest pretrain loss: {best_loss:.4f}")
    print(f"Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
