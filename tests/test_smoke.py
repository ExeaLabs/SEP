"""Smoke tests for the SEP training pipeline."""

import os
import subprocess
import sys


def test_mock_data_generation():
    """Mock mode should create metadata and HDF5 files."""
    subprocess.run(
        [sys.executable, "main.py", "--mode", "mock"],
        check=True,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    assert os.path.exists("metadata_clean.csv")
    assert os.path.exists("modalities.hdf5")


def test_dataset_loads():
    """Dataset should load samples from mock artefacts."""
    from data.dataset import SpeciesExtinctionDataset

    ds = SpeciesExtinctionDataset()
    assert len(ds) > 0
    sat, clim, label = ds[0]
    assert sat.shape == (3, 224, 224)
    assert clim.shape == (30, 12)
    assert 0 <= label.item() <= 4
