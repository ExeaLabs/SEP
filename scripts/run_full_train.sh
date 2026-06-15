#!/usr/bin/env bash
set -e

echo "=== Running SEP Full Scale Training ==="

# Check if real Kaggle data exists; download if not
if ! ls *iucn*.csv *IUCN*.csv *extinction*.csv 2>/dev/null; then
    echo "IUCN data not found. Initiating Kaggle download..."
    bash scripts/download_datasets.sh
fi

# Prepare merged metadata + HDF5 modalities
echo "Preparing datasets..."
python data/data_prep.py

# Train the model
echo "Training model..."
python main.py --mode train --epochs 50 --batch_size 16

# Evaluate
echo "Evaluating model..."
python main.py --mode evaluate

echo "=== Full Scale Training Complete ==="
