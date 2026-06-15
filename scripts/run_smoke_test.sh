#!/usr/bin/env bash
set -e

echo "=== Running SEP Smoke Test ==="

# 1. Generate synthetic mock data
python main.py --mode mock

# 2. Train and evaluate (2 epochs for speed)
python main.py --mode train --epochs 2 --batch_size 8
python main.py --mode evaluate --batch_size 8

echo "=== Smoke Test Complete ==="
