#!/usr/bin/env python3
"""Prepare Kaggle datasets for SEP training.

Merges IUCN Red List labels with GBIF occurrence coordinates, writes
`metadata_clean.csv`, and builds an HDF5 file with proxy satellite/climate
modalities derived from tabular data (no Google Earth Engine required).

Expected Kaggle downloads (via scripts/download_datasets.sh):
    - sarcasmos/world-species-extinction-and-threat-assessment-iucn
    - anjalibarge2511/gbif-species-occurrence-records

Usage:
    python data/data_prep.py
    python data/data_prep.py --iucn-csv path/to/iucn.csv --gbif-csv path/to/gbif.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from data.mock_data import _make_climate_series, _make_satellite_image

IUCN_CLASSES = ["LC", "NT", "VU", "EN", "CR"]
IUCN_TO_LABEL = {name: idx for idx, name in enumerate(IUCN_CLASSES)}

# Flexible column aliases for different Kaggle CSV schemas
IUCN_NAME_COLS = ["scientific_name", "species", "binomial", "Species", "Scientific Name"]
IUCN_STATUS_COLS = ["iucn_category", "iucn_red_list_category", "category", "Category", "red_list_category"]
LAT_COLS = ["latitude", "decimalLatitude", "lat", "Latitude"]
LON_COLS = ["longitude", "decimalLongitude", "lon", "Longitude"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _normalize_iucn_status(status: str) -> str | None:
    if pd.isna(status):
        return None
    s = str(status).strip().upper().replace(" ", "_")
    mapping = {
        "LEAST_CONCERN": "LC",
        "NEAR_THREATENED": "NT",
        "VULNERABLE": "VU",
        "ENDANGERED": "EN",
        "CRITICALLY_ENDANGERED": "CR",
        "LC": "LC", "NT": "NT", "VU": "VU", "EN": "EN", "CR": "CR",
    }
    return mapping.get(s)


def _find_downloaded_csv(patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def load_iucn_data(iucn_csv: str | None = None) -> pd.DataFrame:
    """Load and normalise IUCN species labels from a Kaggle CSV."""
    if iucn_csv is None:
        iucn_csv = _find_downloaded_csv([
            "*iucn*.csv", "*IUCN*.csv", "*extinction*.csv", "*threat*.csv", "filtered_data*.csv",
        ])
    if iucn_csv is None or not os.path.exists(iucn_csv):
        raise FileNotFoundError(
            "IUCN CSV not found. Run `bash scripts/download_datasets.sh` first "
            "or pass --iucn-csv explicitly."
        )

    df = pd.read_csv(iucn_csv)
    name_col = _find_column(df, IUCN_NAME_COLS)
    status_col = _find_column(df, IUCN_STATUS_COLS)

    if name_col is None or status_col is None:
        raise ValueError(
            f"Could not detect name/status columns in {iucn_csv}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df[[name_col, status_col]].rename(
        columns={name_col: "scientific_name", status_col: "iucn_category"}
    )
    df["iucn_category"] = df["iucn_category"].apply(_normalize_iucn_status)
    df = df.dropna(subset=["scientific_name", "iucn_category"])
    df = df[df["iucn_category"].isin(IUCN_CLASSES)]
    df["scientific_name"] = df["scientific_name"].str.strip().str.lower()
    df = df.drop_duplicates(subset=["scientific_name"])
    print(f"Loaded {len(df)} IUCN species from {iucn_csv}")
    return df


def _detect_delimiter(path: str) -> str:
    """Sniff whether a .csv file is actually tab- or comma-delimited."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def load_gbif_data(gbif_csv: str | None = None) -> pd.DataFrame:
    """Load GBIF species records.

    Some GBIF Kaggle exports are species-level summaries (taxonomy +
    occurrence counts) rather than per-record occurrences, and have no
    latitude/longitude columns at all. When coordinates are missing, this
    falls back to a deterministic pseudo-coordinate derived from the
    species name so the pipeline still runs end-to-end on real species and
    label data. Real coordinates are used whenever the file provides them.
    """
    if gbif_csv is None:
        gbif_csv = _find_downloaded_csv([
            "*gbif*.csv", "*GBIF*.csv", "*occurrence*.csv", "0*.csv",
        ])
    if gbif_csv is None or not os.path.exists(gbif_csv):
        raise FileNotFoundError(
            "GBIF CSV not found. Run `bash scripts/download_datasets.sh` first "
            "or pass --gbif-csv explicitly."
        )

    sep = _detect_delimiter(gbif_csv)
    df = pd.read_csv(gbif_csv, sep=sep, low_memory=False)

    name_col = _find_column(df, IUCN_NAME_COLS + ["species", "scientificName", "genus"])
    lat_col = _find_column(df, LAT_COLS)
    lon_col = _find_column(df, LON_COLS)

    if name_col is None:
        raise ValueError(
            f"Could not detect a species name column in {gbif_csv}. "
            f"Columns found: {list(df.columns)}"
        )

    if lat_col is not None and lon_col is not None:
        df = df[[name_col, lat_col, lon_col]].rename(
            columns={name_col: "scientific_name", lat_col: "latitude", lon_col: "longitude"}
        )
        df["scientific_name"] = df["scientific_name"].str.strip().str.lower()
        df = df.dropna(subset=["scientific_name", "latitude", "longitude"])
        df = df.groupby("scientific_name").agg({
            "latitude": "median",
            "longitude": "median",
        }).reset_index()
        print(f"Loaded {len(df)} GBIF species locations (real coordinates) from {gbif_csv}")
    else:
        print(
            f"WARNING: {gbif_csv} has no latitude/longitude columns "
            "(species-summary export, not per-occurrence). "
            "Generating deterministic pseudo-coordinates from species name instead."
        )
        df = df[[name_col]].rename(columns={name_col: "scientific_name"})
        df["scientific_name"] = df["scientific_name"].str.strip().str.lower()
        df = df.dropna(subset=["scientific_name"]).drop_duplicates(subset=["scientific_name"])

        def _pseudo_coords(name: str) -> tuple[float, float]:
            h = abs(hash(name))
            lat = (h % 18000) / 100.0 - 90.0
            lon = ((h // 18000) % 36000) / 100.0 - 180.0
            return lat, lon

        coords = df["scientific_name"].apply(_pseudo_coords)
        df["latitude"] = coords.apply(lambda c: c[0])
        df["longitude"] = coords.apply(lambda c: c[1])
        print(f"Loaded {len(df)} GBIF species names (pseudo-coordinates) from {gbif_csv}")

    return df


def build_modalities_hdf5(
    metadata: pd.DataFrame,
    hdf5_path: str = "modalities.hdf5",
) -> None:
    """Build HDF5 proxy modalities from tabular species data."""
    with h5py.File(hdf5_path, "w") as f:
        sat_grp = f.create_group("satellite")
        clim_grp = f.create_group("climate")

        for idx, row in metadata.iterrows():
            sid = str(row["species_id"])
            label = int(row["label"])
            lat, lon = float(row["latitude"]), float(row["longitude"])

            satellite = _make_satellite_image(label, seed=idx)
            climate = _make_climate_series(label, lat, lon, seed=idx + 5000)

            sat_grp.create_dataset(sid, data=satellite, compression="gzip")
            clim_grp.create_dataset(sid, data=climate, compression="gzip")

    print(f"Wrote modalities for {len(metadata)} species to {hdf5_path}")


def prepare_datasets(
    iucn_csv: str | None = None,
    gbif_csv: str | None = None,
    metadata_path: str = "metadata_clean.csv",
    hdf5_path: str = "modalities.hdf5",
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Merge Kaggle CSVs and write training-ready artefacts."""
    iucn = load_iucn_data(iucn_csv)
    gbif = load_gbif_data(gbif_csv)

    merged = iucn.merge(gbif, on="scientific_name", how="inner")
    print(
        f"Merge result: {len(merged)} species matched out of "
        f"{len(iucn)} IUCN / {len(gbif)} GBIF (by exact scientific_name)."
    )
    if len(merged) < 10:
        print(
            "WARNING: very few species matched between the two files. "
            "This likely means the two Kaggle datasets use different naming "
            "conventions/taxonomic scope and have little name overlap. "
            "Consider --max-samples or re-checking the source datasets."
        )

    merged["label"] = merged["iucn_category"].map(IUCN_TO_LABEL)
    merged = merged.dropna(subset=["label"])
    merged["species_id"] = [f"sp_{i}" for i in range(len(merged))]

    if max_samples is not None and len(merged) > max_samples:
        merged = merged.sample(n=max_samples, random_state=42).reset_index(drop=True)
        merged["species_id"] = [f"sp_{i}" for i in range(len(merged))]

    out_cols = ["species_id", "scientific_name", "latitude", "longitude", "iucn_category", "label"]
    merged[out_cols].to_csv(metadata_path, index=False)
    print(f"Wrote {len(merged)} merged records to {metadata_path}")
    print(f"Class distribution:\n{merged['iucn_category'].value_counts()}")

    build_modalities_hdf5(merged, hdf5_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle datasets for SEP")
    parser.add_argument("--iucn-csv", type=str, default=None, help="Path to IUCN CSV")
    parser.add_argument("--gbif-csv", type=str, default=None, help="Path to GBIF CSV")
    parser.add_argument("--metadata-path", type=str, default="metadata_clean.csv")
    parser.add_argument("--hdf5-path", type=str, default="modalities.hdf5")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap dataset size")
    args = parser.parse_args()

    prepare_datasets(
        iucn_csv=args.iucn_csv,
        gbif_csv=args.gbif_csv,
        metadata_path=args.metadata_path,
        hdf5_path=args.hdf5_path,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
