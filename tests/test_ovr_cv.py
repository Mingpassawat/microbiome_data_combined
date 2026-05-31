from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "experiments" / "phase1_biomegpt_baseline"
sys.path.insert(0, str(PHASE1))

from ovr_cv import make_ovr_folds, summarize_metric_dicts  # noqa: E402


class OvrCvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        meta_path = ROOT / "data" / "combined_microbiome" / "metadata.csv"
        splits_path = ROOT / "data" / "combined_microbiome" / "task_splits.json"
        cls.meta = pd.read_csv(meta_path, low_memory=False)
        if "sample_key.1" in cls.meta.columns:
            cls.meta = cls.meta.drop(columns=["sample_key.1"])
        cls.meta = cls.meta.set_index("sample_key")
        cls.meta = cls.meta[~cls.meta.index.duplicated(keep="first")]
        with open(splits_path) as f:
            cls.splits = json.load(f)

    def labels_for(self, disease: str) -> np.ndarray:
        task = self.splits[disease]
        keys = [k for k in task["train_keys"] if k in self.meta.index]
        return (self.meta.loc[keys, "label"] == disease).astype(int).to_numpy()

    def test_real_ovr_tasks_have_two_class_stratified_folds(self) -> None:
        for disease in ["Colorectal cancer", "IBD", "Obesity", "Type 2 diabetes", "Liver Cirrhosis"]:
            with self.subTest(disease=disease):
                y = self.labels_for(disease)
                folds, meta = make_ovr_folds(y, requested_folds=10, seed=67)
                self.assertEqual(meta["cv_status"], "available")
                self.assertEqual(meta["actual_folds"], 10)
                for _, va_idx in folds:
                    self.assertGreater(int(y[va_idx].sum()), 0)
                    self.assertGreater(int((y[va_idx] == 0).sum()), 0)

    def test_supported_ovr_tasks_do_not_emit_undefined_auroc_warnings(self) -> None:
        import warnings

        for disease in ["Colorectal cancer", "IBD", "Obesity", "Type 2 diabetes", "Liver Cirrhosis"]:
            with self.subTest(disease=disease):
                y = self.labels_for(disease)
                folds, _ = make_ovr_folds(y, requested_folds=10, seed=67)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for _, va_idx in folds:
                        probs = np.linspace(0.0, 1.0, num=len(va_idx))
                        roc_auc_score(y[va_idx], probs)
                messages = [str(item.message) for item in caught]
                self.assertFalse(any("Only one class is present" in msg for msg in messages))

    def test_liver_cirrhosis_has_expected_positive_counts(self) -> None:
        y = self.labels_for("Liver Cirrhosis")
        folds, _ = make_ovr_folds(y, requested_folds=10, seed=67)
        counts = sorted(int(y[va_idx].sum()) for _, va_idx in folds)
        self.assertEqual(min(counts), 12)
        self.assertEqual(max(counts), 13)
        self.assertEqual(sum(counts), int(y.sum()))

    def test_holdout_positive_keys_are_excluded_from_internal_cv_training(self) -> None:
        for disease in ["Obesity", "Type 2 diabetes"]:
            with self.subTest(disease=disease):
                task = self.splits[disease]
                train_keys = set(task["train_keys"])
                val_pos_keys = set(task["val_pos_keys"])
                self.assertTrue(val_pos_keys)
                self.assertTrue(train_keys.isdisjoint(val_pos_keys))

    def test_nan_safe_metric_summary(self) -> None:
        summary = summarize_metric_dicts([
            {"auroc": float("nan"), "binary_f1": 0.2, "n_pos": 1, "n_neg": 2},
            {"auroc": 0.8, "binary_f1": 0.6, "n_pos": 1, "n_neg": 2},
        ])
        self.assertAlmostEqual(summary["mean"]["auroc"], 0.8)
        self.assertAlmostEqual(summary["mean"]["binary_f1"], 0.4)
        self.assertNotIn("n_pos", summary["mean"])

    def test_too_few_positives_is_unavailable(self) -> None:
        folds, meta = make_ovr_folds(np.array([1, 0, 0, 0]), requested_folds=10, seed=67)
        self.assertEqual(folds, [])
        self.assertEqual(meta["cv_status"], "unavailable")
        self.assertIn("need at least 2 positives", meta["cv_unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
