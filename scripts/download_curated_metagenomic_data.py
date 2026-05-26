#!/usr/bin/env python3
"""Download curatedMetagenomicData relative-abundance resources without R.

This script fetches the curatedMetagenomicData sample metadata from the
waldronlab GitHub repository and the study-level relative_abundance matrices
from Bioconductor ExperimentHub. It converts MetaPhlAn species-level relative
abundances into Python-friendly CSV files.

Default output is the BiomeGPT-relevant subset: stool samples with a non-null
`disease` label, represented as samples x species.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sqlite3
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rdata

EXPERIMENTHUB_SQLITE = "https://experimenthub.bioconductor.org/metadata/experimenthub.sqlite3"
SAMPLE_METADATA_RDA = "https://raw.githubusercontent.com/waldronlab/curatedMetagenomicData/devel/data/sampleMetadata.rda"
USER_AGENT = "Microbiome_transformer curatedMetagenomicData downloader"


def urlretrieve(url: str, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
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


def read_rda(path: Path) -> dict:
    parsed = rdata.parser.parse_file(path)
    return rdata.conversion.convert(parsed)


def latest_relative_abundance_resources(sqlite_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(sqlite_path)
    query = """
        SELECT
            r.ah_id,
            r.title,
            lp.location_prefix || rp.rdatapath AS url,
            rp.rdatapath
        FROM resources r
        JOIN rdatapaths rp ON rp.resource_id = r.id
        JOIN location_prefixes lp ON lp.id = r.location_prefix_id
        WHERE r.title LIKE '%.relative_abundance'
    """
    rows = pd.read_sql_query(query, con)
    con.close()
    parts = rows["title"].str.split(".", expand=True)
    rows["date_added"] = parts[0]
    rows["study_name"] = parts[1]
    rows["data_type"] = parts[2]
    rows = rows.sort_values(["study_name", "data_type", "date_added"])
    rows = rows.groupby(["study_name", "data_type"], as_index=False).tail(1)
    return rows.sort_values("study_name").reset_index(drop=True)


def make_sample_keys(meta: pd.DataFrame) -> pd.Series:
    counts = meta["sample_id"].value_counts(dropna=False)
    duplicate = meta["sample_id"].map(counts).fillna(0).gt(1)
    return meta["sample_id"].where(~duplicate, meta["sample_id"].astype(str) + "." + meta["study_name"].astype(str))


def is_species_feature(name: str) -> bool:
    last = str(name).split("|")[-1]
    return last.startswith("s__")


def species_short_name(name: str) -> str:
    last = str(name).split("|")[-1]
    if last.startswith("s__"):
        return last[3:].replace("_", " ")
    return last


def open_csv_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="")
    return path.open("w", newline="")


def write_outputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = cache_dir / "experimenthub.sqlite3"
    metadata_rda = cache_dir / "sampleMetadata.rda"
    print("Downloading metadata indexes...")
    urlretrieve(EXPERIMENTHUB_SQLITE, sqlite_path)
    urlretrieve(SAMPLE_METADATA_RDA, metadata_rda)

    sample_metadata = read_rda(metadata_rda)["sampleMetadata"].copy()
    sample_metadata["sample_key"] = make_sample_keys(sample_metadata)

    selected = sample_metadata.copy()
    if args.body_site:
        selected = selected[selected["body_site"].astype("string") == args.body_site]
    if args.require_disease:
        disease = selected["disease"].astype("string")
        selected = selected[disease.notna() & disease.ne("<NA>") & disease.ne("")]

    selected_ids_by_study = {
        study: set(group["sample_id"].astype(str))
        for study, group in selected.groupby("study_name", dropna=False)
    }
    key_by_study_sample = {
        (row.study_name, row.sample_id): row.sample_key
        for row in selected[["study_name", "sample_id", "sample_key"]].itertuples(index=False)
    }

    resources = latest_relative_abundance_resources(sqlite_path)
    resources = resources[resources["study_name"].isin(selected_ids_by_study)].reset_index(drop=True)
    if args.max_studies:
        resources = resources.head(args.max_studies)

    print(f"Selected {len(selected):,} metadata rows across {len(resources):,} studies.")

    all_frames: list[pd.DataFrame] = []
    feature_counter: Counter[str] = Counter()
    manifest_rows = []
    observed_sample_keys: list[str] = []
    observed_sample_key_set: set[str] = set()

    sparse_path = out_dir / args.sparse_csv
    sparse_handle = open_csv_maybe_gzip(sparse_path)
    sparse_writer = csv.writer(sparse_handle)
    sparse_writer.writerow(["sample_key", "species", "relative_abundance"])

    try:
        for idx, row in resources.iterrows():
            title = row["title"]
            study = row["study_name"]
            rda_path = cache_dir / "resources" / row["rdatapath"]
            print(f"[{idx + 1:03d}/{len(resources):03d}] {title}")
            urlretrieve(row["url"], rda_path)
            obj = read_rda(rda_path)[title]

            taxa = pd.Index(obj.coords[obj.dims[0]].values.astype(str), name="species")
            samples = pd.Index(obj.coords[obj.dims[1]].values.astype(str), name="sample_id")
            keep_taxa = np.fromiter((is_species_feature(t) for t in taxa), dtype=bool, count=len(taxa))
            wanted_samples = selected_ids_by_study.get(study, set())
            keep_samples = np.fromiter((s in wanted_samples for s in samples), dtype=bool, count=len(samples))

            n_samples = int(keep_samples.sum())
            n_species = int(keep_taxa.sum())
            manifest_rows.append({
                "ah_id": row["ah_id"],
                "title": title,
                "study_name": study,
                "resource_url": row["url"],
                "selected_samples": n_samples,
                "species_features": n_species,
            })
            if n_samples == 0 or n_species == 0:
                continue

            arr = np.asarray(obj.values)[keep_taxa][:, keep_samples]
            taxa_kept = taxa[keep_taxa]
            samples_kept = samples[keep_samples]
            sample_keys = [key_by_study_sample[(study, sample)] for sample in samples_kept]
            for sample_key in sample_keys:
                if sample_key not in observed_sample_key_set:
                    observed_sample_key_set.add(sample_key)
                    observed_sample_keys.append(sample_key)
            feature_counter.update(map(str, taxa_kept))

            # Sparse nonzero CSV is much smaller and still directly readable by pandas.
            nz_taxa, nz_samples = np.nonzero(arr)
            for tax_i, sample_i in zip(nz_taxa, nz_samples):
                val = float(arr[tax_i, sample_i])
                if val != 0.0:
                    sparse_writer.writerow([sample_keys[sample_i], taxa_kept[tax_i], f"{val:.12g}"])

            if args.write_wide:
                frame = pd.DataFrame(arr.T, index=sample_keys, columns=taxa_kept)
                all_frames.append(frame)
    finally:
        sparse_handle.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "resource_manifest.csv", index=False)

    selected = selected.set_index("sample_key", drop=False)
    selected_matched = selected.loc[observed_sample_keys]
    selected_matched.to_csv(out_dir / "metadata_stool_labeled.csv", index=True)
    missing_metadata = selected.loc[selected.index.difference(observed_sample_keys)]
    missing_metadata.to_csv(out_dir / "metadata_stool_labeled_missing_assay.csv", index=True)
    sample_metadata.set_index("sample_key", drop=False).to_csv(out_dir / "metadata_all.csv", index=True)

    features = pd.DataFrame({"species": sorted(feature_counter)})
    if not features.empty:
        features["species_short"] = features["species"].map(species_short_name)
        features.to_csv(out_dir / "species_features.csv", index=False)

    if args.write_wide:
        print("Merging study matrices into one wide samples x species table...")
        wide = pd.concat(all_frames, axis=0, join="outer").fillna(0.0)
        wide.index.name = "sample_key"
        wide = wide.loc[observed_sample_keys]
        wide.to_csv(out_dir / args.wide_csv)
        print(f"Wide matrix: {wide.shape[0]:,} samples x {wide.shape[1]:,} species")

    print("Done.")
    print(f"Metadata subset: {out_dir / 'metadata_stool_labeled.csv'}")
    print(f"Sparse abundances: {sparse_path}")
    if args.write_wide:
        print(f"Wide abundances: {out_dir / args.wide_csv}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/curated_metagenomic_data")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--body-site", default="stool", help="Filter sampleMetadata body_site; use '' for all body sites.")
    parser.add_argument("--require-disease", action="store_true", default=True, help="Keep only samples with non-null disease labels.")
    parser.add_argument("--no-require-disease", dest="require_disease", action="store_false")
    parser.add_argument("--write-wide", action="store_true", help="Also write one dense samples x species CSV.")
    parser.add_argument("--wide-csv", default="relative_abundance_species_wide.csv")
    parser.add_argument("--sparse-csv", default="relative_abundance_species_sparse.csv.gz")
    parser.add_argument("--max-studies", type=int, default=0, help="Debug: limit number of studies.")
    args = parser.parse_args(list(argv))
    if args.body_site == "":
        args.body_site = None
    return args


if __name__ == "__main__":
    write_outputs(parse_args(sys.argv[1:]))
