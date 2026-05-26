#!/usr/bin/env python3
"""Combine cMD, GMHI, and GMWI2 species abundance CSVs for Python training.

The three sources use different MetaPhlAn versions and slightly different
feature names. This script aligns features by the terminal species token:

    k__...|g__Escherichia|s__Escherichia_coli -> s__Escherichia_coli
    s__Escherichia_coli                       -> s__Escherichia_coli

The default output uses the union of species and fills missing species with 0.
Use ``--vocab intersection`` if you want only species present in all sources.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "cmd": {
        "dir": "curated_metagenomic_data",
        "metadata": "metadata_stool_labeled.csv",
        "label": "disease",
        "study": "study_name",
        "sample": "sample_id",
    },
    "gmhi": {
        "dir": "gmhi",
        "metadata": "metadata.csv",
        "label": "phenotype",
        "study": "study_phenotype",
        "sample": "sample_accession_or_sample_id",
    },
    "gmwi2": {
        "dir": "gmwi2",
        "metadata": "metadata.csv",
        "label": "phenotype",
        "study": "study_id",
        "sample": "sample_accession",
    },
}


def canonical_species(feature: str) -> str | None:
    """Return the canonical species token, or None for non-species features."""
    token = str(feature).split("|")[-1]
    if not token.startswith("s__"):
        return None
    return token


def readable_species(feature: str) -> str:
    token = canonical_species(feature) or str(feature)
    return token.removeprefix("s__").replace("_", " ")


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    replacements = {
        "Crohns disease": "Crohn's Disease",
        "Crohn's disease": "Crohn's Disease",
        "T2D": "Type 2 diabetes",
        "CRC": "Colorectal cancer",
        "ACVD": "Atherosclerotic cardiovascular disease",
        "healthy": "Healthy",
    }
    return replacements.get(text, text)


def is_healthy_label(value: object) -> bool:
    return normalize_label(value).casefold() == "healthy"


def stable_group_split(group: str, val_fraction: float) -> str:
    digest = hashlib.sha1(group.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def read_metadata(path: Path, source: str, spec: dict[str, str]) -> pd.DataFrame:
    meta = pd.read_csv(path, low_memory=False)
    if "sample_key" not in meta.columns:
        raise ValueError(f"{path} does not contain a sample_key column")

    # Some source files contain duplicate sample_key columns from index exports.
    meta = meta.loc[:, ~meta.columns.duplicated()]
    original_sample_key = meta["sample_key"].astype(str)
    label_col = spec["label"]
    study_col = spec["study"]
    sample_col = spec["sample"]

    clean = pd.DataFrame({
        "sample_key": source + ":" + original_sample_key,
        "source": source,
        "original_sample_key": original_sample_key,
        "sample_id": meta[sample_col].astype(str) if sample_col in meta.columns else original_sample_key,
        "study_id": meta[study_col].astype(str) if study_col in meta.columns else source,
        "label": meta[label_col].map(normalize_label),
    })
    clean["is_healthy"] = clean["label"].map(is_healthy_label)

    for optional in ["age", "age_years", "sex", "gender", "country", "continent", "body_site"]:
        if optional in meta.columns:
            clean[optional] = meta[optional]

    return clean.set_index("sample_key", drop=False)


def read_abundance(path: Path, source: str) -> tuple[pd.DataFrame, dict[str, str]]:
    wide = pd.read_csv(path, index_col="sample_key")
    rename: dict[str, str] = {}
    dropped = []
    for column in wide.columns:
        canonical = canonical_species(column)
        if canonical is None:
            dropped.append(column)
        else:
            rename[column] = canonical

    wide = wide[list(rename)].rename(columns=rename)
    if wide.columns.duplicated().any():
        wide = wide.T.groupby(level=0).sum().T

    wide.index = source + ":" + wide.index.astype(str)
    wide.index.name = "sample_key"
    return wide, rename


def write_sparse(wide: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_key", "species", "relative_abundance"])
        for sample_key, row in wide.iterrows():
            nonzero = row[row != 0]
            for species, value in nonzero.items():
                writer.writerow([sample_key, species, f"{float(value):.12g}"])


def row_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert each sample to relative-abundance proportions summing to 1."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)


def bin_abundances(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """BiomeGPT-style per-sample nonzero quantile bins; zeros stay 0."""
    out = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=np.uint8)
    for sample_key, row in df.iterrows():
        nonzero = row[row > 0]
        if nonzero.empty:
            continue
        ranks = nonzero.rank(method="first")
        bins = np.ceil(ranks / len(nonzero) * n_bins).astype(np.uint8)
        out.loc[sample_key, nonzero.index] = bins
    return out


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="data/combined_microbiome")
    parser.add_argument("--vocab", choices=["union", "intersection"], default="union")
    parser.add_argument("--val-fraction", type=float, default=0.07)
    parser.add_argument("--write-sparse", action="store_true", default=True)
    parser.add_argument("--no-write-sparse", dest="write_sparse", action="store_false")
    parser.add_argument("--write-binned", action="store_true", help="Also write BiomeGPT-style 0-100 abundance bins.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrices: dict[str, pd.DataFrame] = {}
    metadata_frames: list[pd.DataFrame] = []
    source_features: dict[str, list[str]] = {}
    feature_maps: dict[str, dict[str, str]] = {}

    for source, spec in DATASETS.items():
        ds_dir = data_root / spec["dir"]
        matrix_path = ds_dir / "relative_abundance_species_wide.csv"
        metadata_path = ds_dir / spec["metadata"]
        print(f"Loading {source}: {matrix_path}")
        matrix, feature_map = read_abundance(matrix_path, source)
        metadata = read_metadata(metadata_path, source, spec)

        shared_index = metadata.index.intersection(matrix.index)
        matrix = matrix.loc[shared_index]
        metadata = metadata.loc[shared_index]

        matrices[source] = matrix
        metadata_frames.append(metadata)
        source_features[source] = sorted(matrix.columns)
        feature_maps[source] = feature_map
        print(f"  {source}: {matrix.shape[0]:,} samples x {matrix.shape[1]:,} species")

    if args.vocab == "union":
        vocabulary = sorted(set().union(*(set(cols) for cols in source_features.values())))
    else:
        vocabulary = sorted(set.intersection(*(set(cols) for cols in source_features.values())))

    aligned = []
    for source, matrix in matrices.items():
        aligned.append(matrix.reindex(columns=vocabulary, fill_value=0.0))
    combined = pd.concat(aligned, axis=0)
    combined.index.name = "sample_key"
    proportions = row_normalize(combined)

    metadata = pd.concat(metadata_frames, axis=0)
    metadata = metadata.loc[combined.index]
    metadata["group_id"] = metadata["source"] + ":" + metadata["study_id"].astype(str)
    metadata["split"] = metadata["group_id"].map(lambda group: stable_group_split(group, args.val_fraction))

    combined.to_csv(output_dir / "relative_abundance_species_wide.csv")
    proportions.to_csv(output_dir / "relative_abundance_species_proportions_wide.csv")
    metadata.to_csv(output_dir / "metadata.csv")
    pd.DataFrame({
        "species": vocabulary,
        "species_short": [readable_species(v) for v in vocabulary],
    }).to_csv(output_dir / "species_features.csv", index=False)

    if args.write_sparse:
        write_sparse(combined, output_dir / "relative_abundance_species_sparse.csv.gz")
        write_sparse(proportions, output_dir / "relative_abundance_species_proportions_sparse.csv.gz")

    if args.write_binned:
        binned = bin_abundances(proportions, n_bins=100)
        binned.to_csv(output_dir / "abundance_bins_100_wide.csv")

    summary = {
        "vocab_mode": args.vocab,
        "n_samples": int(combined.shape[0]),
        "n_species": int(combined.shape[1]),
        "sources": {
            source: {
                "n_samples": int(matrices[source].shape[0]),
                "n_species_before_alignment": int(matrices[source].shape[1]),
            }
            for source in DATASETS
        },
        "label_counts": metadata["label"].value_counts().to_dict(),
        "split_counts": metadata["split"].value_counts().to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "README.md").write_text(
        "# Combined microbiome dataset\n\n"
        "Combined cMD, GMHI, and GMWI2 species-level relative abundances.\n\n"
        f"- Vocabulary mode: `{args.vocab}`\n"
        f"- Samples: `{combined.shape[0]:,}`\n"
        f"- Species features: `{combined.shape[1]:,}`\n"
        "- Sample IDs are source-prefixed as `cmd:<id>`, `gmhi:<id>`, and `gmwi2:<id>`.\n"
        "- Features are aligned by canonical MetaPhlAn species token, for example `s__Escherichia_coli`.\n"
        "- `relative_abundance_species_proportions_wide.csv` is row-normalized to 0-1 proportions and is the recommended training matrix.\n"
        "- `relative_abundance_species_wide.csv` preserves source scale for audit only; cMD/GMHI are near 100 and GMWI2 is near 1.\n"
        "- `metadata.csv` includes `source`, `label`, `is_healthy`, `group_id`, and a deterministic study-level `split`.\n\n"
        "Load with:\n\n"
        "```python\n"
        "import pandas as pd\n\n"
        "base = 'data/combined_microbiome'\n"
        "X = pd.read_csv(f'{base}/relative_abundance_species_proportions_wide.csv', index_col='sample_key')\n"
        "meta = pd.read_csv(f'{base}/metadata.csv', index_col='sample_key')\n"
        "y = meta.loc[X.index, 'label']\n"
        "```\n"
    )

    print(f"Combined: {combined.shape[0]:,} samples x {combined.shape[1]:,} species")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
