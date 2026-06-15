#!/usr/bin/env python3
"""SEP — Species Extinction Risk Predictor.

Central CLI entry point (mirrors EPGNN's main.py).

Usage:
    python main.py --mode mock                          # generate synthetic data
    python main.py --mode train --epochs 10             # train model
    python main.py --mode evaluate                      # evaluate model
    python main.py --mode smoke                         # mock + train + evaluate
    python main.py --mode prep                          # prepare Kaggle datasets
"""

from __future__ import annotations

import argparse
import os
import sys

from engine.trainer import train_model
from engine.evaluator import evaluate_model
from data.mock_data import create_mock_species_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEP - Species Extinction Risk Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["mock", "prep", "train", "evaluate", "smoke"],
        help="mock: generate synthetic data | prep: prepare Kaggle CSVs | "
             "train | evaluate | smoke: run all three",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--metadata_path", type=str, default="metadata_clean.csv",
        help="Path to cleaned metadata CSV",
    )
    parser.add_argument(
        "--hdf5_path", type=str, default="modalities.hdf5",
        help="Path to HDF5 modality store",
    )
    parser.add_argument(
        "--model_path", type=str, default="extinction_model.pth",
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", default=True,
        help="Freeze ResNet backbone during training",
    )
    args = parser.parse_args()

    if args.mode in ("mock", "smoke"):
        print("=== Generating Mock Data ===")
        create_mock_species_data(
            metadata_path=args.metadata_path,
            hdf5_path=args.hdf5_path,
        )

    if args.mode == "prep":
        print("=== Preparing Kaggle Datasets ===")
        from data.data_prep import prepare_datasets
        prepare_datasets(
            metadata_path=args.metadata_path,
            hdf5_path=args.hdf5_path,
        )

    if args.mode in ("train", "smoke"):
        print("=== Starting Training ===")
        if not os.path.exists(args.metadata_path):
            print(
                f"Warning: {args.metadata_path} not found. "
                "Run `python main.py --mode mock` or `python data/data_prep.py` first."
            )
        else:
            train_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                metadata_path=args.metadata_path,
                hdf5_path=args.hdf5_path,
                model_path=args.model_path,
                freeze_backbone=args.freeze_backbone,
            )

    if args.mode in ("evaluate", "smoke"):
        print("=== Starting Evaluation ===")
        evaluate_model(
            batch_size=args.batch_size,
            metadata_path=args.metadata_path,
            hdf5_path=args.hdf5_path,
            model_path=args.model_path,
            freeze_backbone=args.freeze_backbone,
        )


if __name__ == "__main__":
    main()
