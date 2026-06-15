# SEP (Species Extinction Risk Predictor)

SEP is a PyTorch research codebase for predicting IUCN extinction risk categories (LC, NT, VU, EN, CR) from multi-modal inputs: satellite imagery and climate time-series.

This repository follows the same training workflow as [EPGNN](https://github.com/Arya-Addagarla/EPGNN): datasets are downloaded from Kaggle, preprocessed into a metadata CSV + HDF5 modality store, and trained via a single `main.py` CLI.

## Quick start

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

## Core commands

```bash
# Synthetic end-to-end smoke test (no Kaggle download needed)
bash scripts/run_smoke_test.sh

# Or step by step:
python main.py --mode mock                    # generate synthetic data
python main.py --mode train --epochs 10       # train model
python main.py --mode evaluate                # evaluate model
python main.py --mode smoke                   # run all three

# Full training on real Kaggle data
bash scripts/run_full_train.sh
```

## Datasets (Kaggle)

Datasets are not committed to the repo. Download them via the Kaggle API:

| Dataset | Kaggle slug | Purpose |
|---------|-------------|---------|
| World Species Extinction & Threat Assessment (IUCN) | `sarcasmos/world-species-extinction-and-threat-assessment-iucn` | IUCN risk labels |
| GBIF Species Occurrence Records | `anjalibarge2511/gbif-species-occurrence-records` | Species coordinates |

```bash
# Requires ~/.kaggle/kaggle.json API credentials
bash scripts/download_datasets.sh
python data/data_prep.py   # merge CSVs → metadata_clean.csv + modalities.hdf5
```

### How data is organized

Like EPGNN's `metadata_clean.csv` + `mock_waveforms.hdf5` pattern:

| File | Description |
|------|-------------|
| `metadata_clean.csv` | One row per species: `species_id`, `scientific_name`, `latitude`, `longitude`, `iucn_category`, `label` |
| `modalities.hdf5` | Lazy-loaded arrays keyed by `species_id`: |
| | `satellite/{species_id}` → `(3, 224, 224)` RGB habitat tile |
| | `climate/{species_id}` → `(30, 12)` 30-year climate series |

The PyTorch `Dataset` reads the CSV at init and loads HDF5 arrays one sample at a time to manage memory.

### Data preparation pipeline

```
1. bash scripts/download_datasets.sh     # Kaggle download
2. python data/data_prep.py              # merge IUCN + GBIF → metadata + HDF5
3. python main.py --mode train           # train
4. python main.py --mode evaluate        # evaluate
```

For local testing without Kaggle:

```
python main.py --mode mock               # generates metadata_clean.csv + modalities.hdf5
```

## Model architecture

```
satellite_image (3×224×224) ──► ResNet-50 + Spatial Attention ──► 512-d features
                                                                        │
climate_series (30×12)      ──► BiLSTM + Temporal Attention    ──► 512-d features
                                                                        │
                                                    concat ──► FC head ──► 5-class logits
```

## Repository layout

```
SEP/
├── main.py              # Central CLI (mock | prep | train | evaluate | smoke)
├── data/
│   ├── dataset.py       # SpeciesExtinctionDataset (CSV + HDF5)
│   ├── mock_data.py     # Synthetic data generator
│   └── data_prep.py     # Kaggle CSV merge & HDF5 builder
├── models/
│   ├── satellite_encoder.py
│   ├── climate_encoder.py
│   ├── fusion.py        # MultimodalFusionNetwork
│   └── model.py
├── engine/
│   ├── trainer.py       # Training loop
│   └── evaluator.py     # Evaluation loop
├── losses/
│   └── focal_loss.py    # Class-imbalanced focal loss
├── scripts/             # setup, download, smoke test, full train
├── configs/default.yaml # Reference configuration
└── requirements.txt
```

## License

MIT
