"""
make_task_splits.py — precompute per-task train/val sample keys.

Writes data/combined_microbiome/task_splits.json which the experiment scripts
(finetune_ovr.py, baselines.py) load at startup instead of computing holdouts
at runtime.

Each task entry stores:
  train_keys    — sample_keys used for training that task
  val_pos_keys  — positive-class sample_keys for external evaluation
  val_neg_keys  — negative-class sample_keys for external evaluation
  val_source    — human-readable description of where val keys come from

For OvR tasks with no val-split samples, a single train study is held out
(largest by disease sample count) and used as external val alongside all
val-split Healthy samples.

Usage:
    cd experiments/phase1_biomegpt_baseline
    python make_task_splits.py
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml

# ── Task definitions ─────────────────────────────────────────────────────────
# type "binary": use is_healthy label; no holdout needed.
# type "ovr":    disease_label vs "Healthy".
#   holdout_study: name of the largest train study to hold out for external val.
#                  Set to None if the disease already appears in the val split,
#                  or if there is only one train study (Liver Cirrhosis).
TASKS: OrderedDict[str, dict] = OrderedDict([
    ("binary", {
        "type": "binary",
        "description": "healthy (1) vs. all-diseased (0)",
    }),
    ("Colorectal cancer", {
        "type": "ovr",
        "disease_label": "Colorectal cancer",
        "description": "Colorectal cancer (1) vs. Healthy (0)",
        "holdout_study": None,   # already in val split (202 samples)
    }),
    ("IBD", {
        "type": "ovr",
        "disease_label": "IBD",
        "description": "IBD (1) vs. Healthy (0)",
        "holdout_study": None,   # already in val split (204 samples)
    }),
    ("Obesity", {
        "type": "ovr",
        "disease_label": "Obesity",
        "description": "Obesity (1) vs. Healthy (0)",
        "holdout_study": "gmhi:V-12_Obesity",   # largest train study (104 samples)
    }),
    ("Type 2 diabetes", {
        "type": "ovr",
        "disease_label": "Type 2 diabetes",
        "description": "Type 2 diabetes (1) vs. Healthy (0)",
        "holdout_study": "cmd:MetaCardis_2020_a",   # largest train study (549 samples)
    }),
    ("Liver Cirrhosis", {
        "type": "ovr",
        "disease_label": "Liver Cirrhosis",
        "description": "Liver Cirrhosis (1) vs. Healthy (0)",
        "holdout_study": None,
        # Only 1 study — hold out a random 20% of disease samples instead.
        # val_neg = val-split Healthy.
        "sample_val_frac": 0.2,
        "sample_val_seed": 42,
    }),
])


def build_splits(meta: pd.DataFrame) -> dict:
    train = meta[meta["split"] == "train"]
    val   = meta[meta["split"] == "val"]
    out: dict[str, dict] = {}

    for task_name, task_def in TASKS.items():
        if task_def["type"] == "binary":
            tr_keys = list(train.index)
            vp_keys = list(val[val["is_healthy"] == 1].index)
            vn_keys = list(val[val["is_healthy"] == 0].index)
            out[task_name] = {
                "description": task_def["description"],
                "val_source": "original val split",
                "n_train_pos": int((train["is_healthy"] == 1).sum()),
                "n_train_neg": int((train["is_healthy"] == 0).sum()),
                "n_val_pos": len(vp_keys),
                "n_val_neg": len(vn_keys),
                "train_keys": tr_keys,
                "val_pos_keys": vp_keys,
                "val_neg_keys": vn_keys,
            }
            continue

        disease = task_def["disease_label"]
        holdout = task_def["holdout_study"]
        n_val_disease = (val["label"] == disease).sum()

        if holdout:
            # Exclude holdout study from OvR train set
            tr_mask = (
                ((train["label"] == disease) | (train["label"] == "Healthy"))
                & (train["group_id"] != holdout)
            )
            train_keys = list(train[tr_mask].index)
            # Positive val: holdout disease samples (still in train split)
            val_pos_keys = list(
                train[(train["label"] == disease) & (train["group_id"] == holdout)].index
            )
            # Negative val: all val-split Healthy samples
            val_neg_keys = list(val[val["label"] == "Healthy"].index)
            val_source = f"holdout:{holdout} + val-split Healthy"

        elif n_val_disease > 0:
            # Disease already in val split — use it directly
            tr_mask = (train["label"] == disease) | (train["label"] == "Healthy")
            train_keys = list(train[tr_mask].index)
            val_pos_keys = list(val[val["label"] == disease].index)
            val_neg_keys = list(val[val["label"] == "Healthy"].index)
            val_source = "original val split"

        elif task_def.get("sample_val_frac"):
            # Only one study — split disease samples at the sample level
            frac = task_def["sample_val_frac"]
            seed = task_def.get("sample_val_seed", 42)
            disease_keys = list(train[train["label"] == disease].index)
            rng = np.random.default_rng(seed)
            rng.shuffle(disease_keys)
            n_val = max(1, round(len(disease_keys) * frac))
            val_pos_keys = disease_keys[-n_val:]
            disease_train_keys = disease_keys[:-n_val]
            healthy_train_keys = list(train[train["label"] == "Healthy"].index)
            train_keys = disease_train_keys + healthy_train_keys
            val_neg_keys = list(val[val["label"] == "Healthy"].index)
            val_source = f"sample split ({frac:.0%} of single study, seed={seed}) + val-split Healthy"

        else:
            # No val samples, no holdout possible
            tr_mask = (train["label"] == disease) | (train["label"] == "Healthy")
            train_keys = list(train[tr_mask].index)
            val_pos_keys = []
            val_neg_keys = []
            val_source = "none"

        tr_sub = train.loc[[k for k in train_keys if k in train.index]]
        n_pos = int((tr_sub["label"] == disease).sum())
        n_neg = int((tr_sub["label"] == "Healthy").sum())

        out[task_name] = {
            "description": task_def["description"],
            "val_source": val_source,
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
            "n_val_pos": len(val_pos_keys),
            "n_val_neg": len(val_neg_keys),
            "train_keys": train_keys,
            "val_pos_keys": val_pos_keys,
            "val_neg_keys": val_neg_keys,
        }

    return out


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    meta_path = os.path.join(base_dir, cfg["data"]["metadata_path"])
    meta = pd.read_csv(meta_path, low_memory=False)
    if "sample_key.1" in meta.columns:
        meta = meta.drop(columns=["sample_key.1"])
    meta = meta.set_index("sample_key")
    meta = meta[~meta.index.duplicated(keep="first")]
    meta["is_healthy"] = (
        meta["is_healthy"].astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    )

    splits = build_splits(meta)

    print(f"{'Task':<25} {'Train+':>8} {'Train-':>8} {'Val+':>6} {'Val-':>6}  Val source")
    print("-" * 90)
    for task_name, info in splits.items():
        print(
            f"{task_name:<25} "
            f"{info.get('n_train_pos', len(info['train_keys'])):>8} "
            f"{info.get('n_train_neg', 0):>8} "
            f"{info['n_val_pos']:>6} "
            f"{info['n_val_neg']:>6}  "
            f"{info['val_source']}"
        )

    out_path = os.path.join(base_dir, cfg["data"]["task_splits_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
