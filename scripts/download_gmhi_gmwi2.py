#!/usr/bin/env python3
"""Download GMHI and GMWI2 processed datasets as Python-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

GMHI_BASE = "https://raw.githubusercontent.com/jaeyunsung/GMHI_2020/master"
GMWI2_BASE = "https://raw.githubusercontent.com/danielchang2002/GMWI2/main"
USER_AGENT = "Microbiome_transformer GMHI/GMWI2 downloader"


def download(url: str, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as response, tmp.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(path)
            return
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return text.strip("_").lower() or "field"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = slug(name)
        idx = seen.get(base, 0)
        seen[base] = idx + 1
        out.append(base if idx == 0 else f"{base}_{idx + 1}")
    return out


def is_species_column(name: str) -> bool:
    return str(name).split("|")[-1].startswith("s__")


def species_short(name: str) -> str:
    last = str(name).split("|")[-1]
    if last.startswith("s__"):
        return last[3:].replace("_", " ")
    return last


def write_sparse(wide: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_key", "species", "relative_abundance"])
        for sample_key, row in wide.iterrows():
            nonzero = row[row != 0]
            for species, value in nonzero.items():
                writer.writerow([sample_key, species, f"{float(value):.12g}"])


def write_features(columns: pd.Index, path: Path) -> None:
    pd.DataFrame({
        "species": list(columns),
        "species_short": [species_short(col) for col in columns],
    }).to_csv(path, index=False)


def export_gmhi(root: Path) -> None:
    out = root / "data" / "gmhi"
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "4347_final_relative_abundances.txt": f"{GMHI_BASE}/4347_final_relative_abundances.txt",
        "Final_metadata_4347.csv": f"{GMHI_BASE}/Final_metadata_4347.csv",
        "ReadMe.md": f"{GMHI_BASE}/ReadMe.md",
        "study_wise_data.txt": f"{GMHI_BASE}/study_wise_data.txt",
    }
    for filename, url in files.items():
        download(url, raw / filename)

    abundance = pd.read_csv(raw / "4347_final_relative_abundances.txt", sep="\t", index_col=0)
    abundance.index.name = "species"
    wide = abundance.T
    wide.index.name = "sample_key"
    wide.to_csv(out / "relative_abundance_species_wide.csv")
    write_sparse(wide, out / "relative_abundance_species_sparse.csv.gz")
    write_features(wide.columns, out / "species_features.csv")

    # GMHI metadata is stored transposed: rows are metadata fields, columns are samples.
    metadata_raw = pd.read_csv(raw / "Final_metadata_4347.csv", header=None, encoding="utf-8-sig", dtype=str)
    field_rows = metadata_raw.iloc[:33].copy()
    fields = unique_names(field_rows.iloc[:, 0].astype(str).tolist())
    fields[0] = "study_phenotype"
    fields[1] = "study_phenotype_duplicate"
    meta = field_rows.iloc[:, 1:].T
    meta.columns = fields
    meta.insert(0, "sample_key", wide.index.to_list())
    meta = meta.set_index("sample_key", drop=False)
    meta.to_csv(out / "metadata.csv")

    (out / "README.md").write_text(
        "# GMHI export\n\n"
        "Processed from jaeyunsung/GMHI_2020.\n\n"
        "- `relative_abundance_species_wide.csv`: 4,347 samples x species.\n"
        "- `metadata.csv`: metadata aligned to the abundance matrix.\n"
        "- `relative_abundance_species_sparse.csv.gz`: nonzero abundances in long CSV form.\n"
        "- `species_features.csv`: species feature list.\n"
        "- `raw/`: downloaded source files.\n\n"
        "Load with:\n\n"
        "```python\n"
        "import pandas as pd\n"
        "base = 'data/gmhi'\n"
        "X = pd.read_csv(f'{base}/relative_abundance_species_wide.csv', index_col='sample_key')\n"
        "meta = pd.read_csv(f'{base}/metadata.csv', index_col='sample_key')\n"
        "y = meta.loc[X.index, 'phenotype']\n"
        "```\n"
    )
    print(f"GMHI: {wide.shape[0]:,} samples x {wide.shape[1]:,} species")


def export_gmwi2(root: Path) -> None:
    out = root / "data" / "gmwi2"
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)

    data_zip = raw / "data.zip"
    download(f"{GMWI2_BASE}/manuscript/data.zip", data_zip)
    with zipfile.ZipFile(data_zip) as archive:
        archive.extractall(raw)
    download(f"{GMWI2_BASE}/manuscript/Test_dataset_metaphlan3.csv", raw / "Test_dataset_metaphlan3.csv")
    download(f"{GMWI2_BASE}/README.md", raw / "GMWI2_README.md")

    training = pd.read_csv(raw / "data" / "training_set.csv", low_memory=False)
    metadata_cols = ["Study_ID", "Sample Accession", "is_healthy", "Age", "Sex", "Continent", "Phenotype"]
    species_cols = [col for col in training.columns if is_species_column(col)]
    meta = training[metadata_cols].copy()
    meta.columns = ["study_id", "sample_accession", "is_healthy", "age", "sex", "continent", "phenotype"]
    meta.insert(0, "sample_key", meta["sample_accession"].astype(str))
    meta = meta.set_index("sample_key", drop=False)
    meta.to_csv(out / "metadata.csv")

    wide = training.set_index("Sample Accession")[species_cols]
    wide.index.name = "sample_key"
    wide.to_csv(out / "relative_abundance_species_wide.csv")
    write_sparse(wide, out / "relative_abundance_species_sparse.csv.gz")
    write_features(wide.columns, out / "species_features.csv")

    # Keep full-taxonomy and longitudinal processed tables too; these are useful audit/source CSVs.
    training.to_csv(out / "training_set_full_taxonomy.csv", index=False)
    longitudinal = pd.read_csv(raw / "data" / "longitudinal_cases.csv", low_memory=False)
    longitudinal.to_csv(out / "longitudinal_cases_full_taxonomy.csv", index=False)

    (out / "README.md").write_text(
        "# GMWI2 export\n\n"
        "Processed from danielchang2002/GMWI2 `manuscript/data.zip`.\n\n"
        "- `relative_abundance_species_wide.csv`: 8,069 samples x species.\n"
        "- `metadata.csv`: metadata aligned to the abundance matrix, including `is_healthy` and `phenotype`.\n"
        "- `relative_abundance_species_sparse.csv.gz`: nonzero species abundances in long CSV form.\n"
        "- `species_features.csv`: species feature list.\n"
        "- `training_set_full_taxonomy.csv`: original training table with all taxonomic ranks and metadata.\n"
        "- `longitudinal_cases_full_taxonomy.csv`: additional longitudinal table from the archive.\n"
        "- `raw/`: downloaded source files.\n\n"
        "Load with:\n\n"
        "```python\n"
        "import pandas as pd\n"
        "base = 'data/gmwi2'\n"
        "X = pd.read_csv(f'{base}/relative_abundance_species_wide.csv', index_col='sample_key')\n"
        "meta = pd.read_csv(f'{base}/metadata.csv', index_col='sample_key')\n"
        "y = meta.loc[X.index, 'phenotype']\n"
        "```\n"
    )
    print(f"GMWI2: {wide.shape[0]:,} samples x {wide.shape[1]:,} species")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root for output files.")
    parser.add_argument("--dataset", choices=["all", "gmhi", "gmwi2"], default="all")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if args.dataset in {"all", "gmhi"}:
        export_gmhi(root)
    if args.dataset in {"all", "gmwi2"}:
        export_gmwi2(root)


if __name__ == "__main__":
    main(sys.argv[1:])
