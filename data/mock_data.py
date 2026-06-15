"""Synthetic species data generator for local smoke tests.

Mirrors EPGNN's `data/mock_data.py`: writes `metadata_clean.csv` and an HDF5
file with modality arrays so the full train/evaluate pipeline can run without
downloading Kaggle datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h5py

IUCN_CLASSES = ["LC", "NT", "VU", "EN", "CR"]
IUCN_TO_LABEL = {name: idx for idx, name in enumerate(IUCN_CLASSES)}

# Class distribution mirrors real IUCN imbalance (mostly Least Concern)
CLASS_COUNTS = {
    "LC": 50,
    "NT": 20,
    "VU": 15,
    "EN": 10,
    "CR": 5,
}


def _make_satellite_image(label: int, seed: int) -> np.ndarray:
    """Generate a synthetic 224x224 RGB satellite tile (CHW)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.8, size=(3, 224, 224)).astype(np.float32)

    # Higher risk categories get more "disturbed" habitat patterns
    disturbance = (label / 4.0) * 0.4
    noise = rng.normal(0, disturbance, size=(3, 224, 224)).astype(np.float32)
    img = np.clip(base + noise, 0.0, 1.0)

    # Add a deforestation-like band for endangered species
    if label >= 3:
        img[:, 80:140, 80:140] *= 0.3

    return img


def _make_climate_series(label: int, lat: float, lon: float, seed: int) -> np.ndarray:
    """Generate a synthetic 30-year x 12-variable climate series."""
    rng = np.random.default_rng(seed)
    years = 30
    variables = 12

    # Base climate from pseudo-geography
    temp_base = 1.0 - abs(lat) / 90.0
    precip_base = abs(lon) / 180.0

    series = np.zeros((years, variables), dtype=np.float32)
    for v in range(variables):
        trend = rng.normal(0, 0.01 * (label + 1), size=years)
        seasonal = 0.1 * np.sin(np.linspace(0, 4 * np.pi, years) + v)
        base = temp_base if v % 2 == 0 else precip_base
        series[:, v] = base + seasonal + np.cumsum(trend)

    return series


def create_mock_species_data(
    metadata_path: str = "metadata_clean.csv",
    hdf5_path: str = "modalities.hdf5",
) -> None:
    """Create synthetic metadata and HDF5 modalities for smoke testing."""
    records = []
    seed = 0

    for category, count in CLASS_COUNTS.items():
        for i in range(count):
            species_id = f"sp_{category.lower()}_{i}"
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            records.append({
                "species_id": species_id,
                "scientific_name": f"Genus species_{category}_{i}",
                "latitude": lat,
                "longitude": lon,
                "iucn_category": category,
                "label": IUCN_TO_LABEL[category],
            })
            seed += 1

    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)
    print(f"Wrote {len(df)} species records to {metadata_path}")

    with h5py.File(hdf5_path, "w") as f:
        sat_grp = f.create_group("satellite")
        clim_grp = f.create_group("climate")

        for idx, row in df.iterrows():
            sid = row["species_id"]
            label = int(row["label"])
            lat, lon = float(row["latitude"]), float(row["longitude"])

            satellite = _make_satellite_image(label, seed=idx)
            climate = _make_climate_series(label, lat, lon, seed=idx + 1000)

            sat_grp.create_dataset(sid, data=satellite, compression="gzip")
            clim_grp.create_dataset(sid, data=climate, compression="gzip")

    print(f"Wrote satellite + climate arrays to {hdf5_path}")
    print(f"Class distribution: {df['iucn_category'].value_counts().to_dict()}")


if __name__ == "__main__":
    create_mock_species_data()
