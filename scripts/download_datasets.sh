#!/usr/bin/env bash
set -e

echo "=== Downloading SEP Datasets from Kaggle ==="

if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: Kaggle API credentials not found."
    echo "Place your API token at ~/.kaggle/kaggle.json"
    echo "Get one at: https://www.kaggle.com/settings/account"
    exit 1
fi

echo "Downloading IUCN Red List species data..."
kaggle datasets download -d sarcasmos/world-species-extinction-and-threat-assessment-iucn --unzip

echo "Downloading GBIF species occurrence records..."
kaggle datasets download -d anjalibarge2511/gbif-species-occurrence-records --unzip

echo "=== Download Complete ==="
echo "Next: python data/data_prep.py  (or  python main.py --mode prep)"
