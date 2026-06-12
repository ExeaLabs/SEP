#!/usr/bin/env python3
"""
Species Extinction Risk Predictor — Inference Script
=====================================================
Generate predictions for a list of species, output risk assessments as CSV,
and flag high-risk species.

Usage:
    python predict.py \\
        --model-path outputs/best_model.pth \\
        --species-csv data/species_input.csv \\
        --output-dir predictions/ \\
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.model import ExtinctionRiskModel
from src.data.dataset import SpeciesInferenceDataset
from src.visualization.interactive_map import create_prediction_map

# ── constants ────────────────────────────────────────────────────────────────
IUCN_CLASSES: List[str] = ["LC", "NT", "VU", "EN", "CR"]
HIGH_RISK_CLASSES = {"EN", "CR"}
HIGH_RISK_CONFIDENCE_THRESHOLD = 0.7

logger = logging.getLogger("predict")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _setup_logging(output_dir: Path) -> None:
    """Initialise logging with file + console handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"predict_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_model(
    model_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> ExtinctionRiskModel:
    """Load a trained model from a checkpoint file.

    Supports both full checkpoint dicts and raw state-dicts.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model_cfg = config.get("model", {})
    model = ExtinctionRiskModel(num_classes=len(IUCN_CLASSES), **model_cfg)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint from epoch %d (metric=%.4f)",
            ckpt.get("epoch", -1),
            ckpt.get("best_metric", 0.0),
        )
    else:
        model.load_state_dict(ckpt)
        logger.info("Loaded raw state-dict from %s", model_path)

    model.to(device)
    model.eval()
    return model


def _predict_batch(
    model: ExtinctionRiskModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on all batches.

    Returns
    -------
    predicted_classes : ndarray of int, shape (N,)
    confidences       : ndarray of float, shape (N,)
    probabilities     : ndarray of float, shape (N, num_classes)
    """
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items() if k != "species_name"}
            logits = model(**inputs)  # (B, C)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    probabilities = np.concatenate(all_probs, axis=0)
    predicted_classes = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    return predicted_classes, confidences, probabilities


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extinction-risk predictions for species.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--species-csv", type=str, required=True,
        help="CSV file with columns: species_name, lat, lon (and optional features).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="predictions/",
        help="Directory for output CSV and map.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--generate-map", action="store_true",
        help="Generate an interactive Folium map of predictions.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (-1 for CPU).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)

    # ── device ───────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ── config & model ───────────────────────────────────────────────────
    cfg = _load_config(args.config)
    model = _load_model(args.model_path, cfg, device)

    # ── input data ───────────────────────────────────────────────────────
    input_csv = Path(args.species_csv)
    if not input_csv.exists():
        logger.error("Species CSV not found: %s", input_csv)
        sys.exit(1)

    species_df = pd.read_csv(input_csv)
    required_cols = {"species_name", "lat", "lon"}
    missing = required_cols - set(species_df.columns)
    if missing:
        logger.error("Missing required columns in CSV: %s", missing)
        sys.exit(1)

    logger.info("Loaded %d species from %s", len(species_df), input_csv)

    # ── dataset / dataloader ─────────────────────────────────────────────
    dataset = SpeciesInferenceDataset(
        dataframe=species_df,
        config=cfg.get("data", {}),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── inference ────────────────────────────────────────────────────────
    logger.info("Running inference on %d species …", len(dataset))
    predicted_classes, confidences, probabilities = _predict_batch(model, dataloader, device)

    # ── build results DataFrame ──────────────────────────────────────────
    results = pd.DataFrame({
        "species_name": species_df["species_name"].values,
        "lat": species_df["lat"].values,
        "lon": species_df["lon"].values,
        "predicted_class": [IUCN_CLASSES[c] for c in predicted_classes],
        "confidence": np.round(confidences, 4),
    })

    # Per-class probabilities
    for i, cls_name in enumerate(IUCN_CLASSES):
        results[f"{cls_name}_prob"] = np.round(probabilities[:, i], 4)

    # ── flag high-risk species ───────────────────────────────────────────
    results["high_risk"] = (
        results["predicted_class"].isin(HIGH_RISK_CLASSES)
        & (results["confidence"] >= HIGH_RISK_CONFIDENCE_THRESHOLD)
    )

    high_risk_count = results["high_risk"].sum()
    logger.info(
        "High-risk species flagged: %d / %d (predicted EN/CR with confidence ≥ %.2f)",
        high_risk_count,
        len(results),
        HIGH_RISK_CONFIDENCE_THRESHOLD,
    )

    # ── save CSV ─────────────────────────────────────────────────────────
    csv_path = output_dir / "predictions.csv"
    results.to_csv(csv_path, index=False)
    logger.info("Predictions saved to %s", csv_path)

    # ── save high-risk subset ────────────────────────────────────────────
    if high_risk_count > 0:
        hr_path = output_dir / "high_risk_species.csv"
        results[results["high_risk"]].to_csv(hr_path, index=False)
        logger.info("High-risk species saved to %s", hr_path)

    # ── interactive map (optional) ───────────────────────────────────────
    if args.generate_map:
        map_path = output_dir / "prediction_map.html"
        create_prediction_map(results, output_path=str(map_path))
        logger.info("Interactive map saved to %s", map_path)

    # ── summary ──────────────────────────────────────────────────────────
    logger.info("─── Prediction Summary ───")
    for cls in IUCN_CLASSES:
        n = (results["predicted_class"] == cls).sum()
        logger.info("  %s: %d species (%.1f%%)", cls, n, 100.0 * n / len(results))
    logger.info("──────────────────────────")
    logger.info("✅ Prediction complete.")


if __name__ == "__main__":
    main()
