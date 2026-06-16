"""PyTorch Dataset for species extinction risk prediction.

Reads species metadata from a cleaned CSV and lazily loads satellite imagery
and climate time-series from an HDF5 store — mirroring the EPGNN pattern of
connecting CSV metadata to binary modality files for VRAM-efficient training.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IUCN_CLASSES = ["LC", "NT", "VU", "EN", "CR"]
IUCN_TO_LABEL = {name: idx for idx, name in enumerate(IUCN_CLASSES)}


class SpeciesExtinctionDataset(Dataset):
    """Multi-modal dataset for IUCN extinction risk classification.

    Expected metadata CSV columns:
        - species_id: unique identifier used as HDF5 key
        - latitude, longitude: species occurrence coordinates
        - iucn_category: one of LC, NT, VU, EN, CR
        - label: integer class index (0-4)

    Expected HDF5 structure:
        satellite/{species_id}  -> float32 array (3, 224, 224)
        climate/{species_id}    -> float32 array (30, 12)

    Args:
        metadata_path: Path to cleaned metadata CSV.
        hdf5_path: Path to HDF5 file with satellite and climate arrays.
        transform: Optional transform applied to satellite images.
    """

    def __init__(
        self,
        metadata_path: str = "metadata_clean.csv",
        hdf5_path: str = "modalities.hdf5",
        transform=None,
    ) -> None:
        self.metadata_path = metadata_path
        self.hdf5_path = hdf5_path
        self.transform = transform

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "Run `python main.py --mode mock` or `python data/data_prep.py` first."
            )
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {hdf5_path}. "
                "Run `python main.py --mode mock` or `python data/data_prep.py` first."
            )

        self.metadata = pd.read_csv(metadata_path)
        required = {"species_id", "label"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

        with h5py.File(self.hdf5_path, "r") as f:
            valid_ids = set(f["satellite"].keys()) & set(f["climate"].keys())

        before = len(self.metadata)
        self.metadata = self.metadata[
            self.metadata["species_id"].astype(str).isin(valid_ids)
        ].reset_index(drop=True)
        dropped = before - len(self.metadata)
        if dropped:
            print(
                f"[SpeciesExtinctionDataset] Warning: dropped {dropped} rows from "
                f"{metadata_path} with no matching entry in {hdf5_path} "
                f"(metadata/HDF5 out of sync — re-run data_prep.py to fix at the source)."
            )

        self.labels = self.metadata["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.metadata)

    def get_labels(self) -> list[int]:
        """Return all integer labels (used for stratified sampling)."""
        return self.labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        species_id = str(row["species_id"])
        label = int(row["label"])

        with h5py.File(self.hdf5_path, "r") as f:
            satellite = self._load_satellite(f, species_id)
            climate = self._load_climate(f, species_id)

        satellite = torch.tensor(satellite, dtype=torch.float32)
        climate = torch.tensor(climate, dtype=torch.float32)
        target = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            satellite = self.transform(satellite)

        return satellite, climate, target

    @staticmethod
    def _load_satellite(f: h5py.File, species_id: str) -> np.ndarray:
        """Load satellite image array in CHW format."""
        for key in (f"satellite/{species_id}", species_id):
            if key in f:
                arr = f[key][:]
                break
        else:
            raise KeyError(f"Satellite data not found for species_id={species_id}")

        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    @staticmethod
    def _load_climate(f: h5py.File, species_id: str) -> np.ndarray:
        """Load climate time-series array (years x variables)."""
        for key in (f"climate/{species_id}", f"climate/{species_id}/data"):
            if key in f:
                return f[key][:].astype(np.float32)
        raise KeyError(f"Climate data not found for species_id={species_id}")


if __name__ == "__main__":
    ds = SpeciesExtinctionDataset()
    sat, clim, y = ds[0]
    print(f"Dataset size: {len(ds)}")
    print(f"Satellite shape: {sat.shape}, Climate shape: {clim.shape}, Label: {y.item()}")
