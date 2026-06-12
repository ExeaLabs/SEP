#!/usr/bin/env python3
"""
Species Extinction Risk Predictor — Training Script
=====================================================
Entry-point for training the multi-modal extinction risk classifier.

Usage:
    python train.py --config configs/default.yaml --data-dir data/ --output-dir outputs/
    python train.py --config configs/default.yaml --resume checkpoints/best.pth --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import SpeciesDataset
from src.models.model import ExtinctionRiskModel
from src.losses.focal_loss import FocalLoss
from src.engine.trainer import Trainer
from src.visualization.plots import plot_training_curves, plot_class_distribution

# ── IUCN risk categories ────────────────────────────────────────────────────
IUCN_CLASSES: List[str] = ["LC", "NT", "VU", "EN", "CR"]

logger = logging.getLogger("train")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    """Configure root logger with both file and console handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger.info("Logging initialised – file: %s", log_path)


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg


def _build_stratified_sampler(dataset: SpeciesDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-balanced mini-batches.

    This ensures that under-represented IUCN categories (EN, CR) are sampled
    proportionally during training, counteracting severe class imbalance.
    """
    labels = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=len(IUCN_CLASSES))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
    logger.info(
        "Stratified sampler created – class counts: %s",
        dict(zip(IUCN_CLASSES, class_counts.tolist())),
    )
    return sampler


def _build_class_weights(dataset: SpeciesDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for the loss function."""
    labels = dataset.get_labels()
    counts = np.bincount(labels, minlength=len(IUCN_CLASSES)).astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(IUCN_CLASSES)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Species Extinction Risk Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/",
        help="Root directory containing processed datasets.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/",
        help="Directory for model checkpoints, logs, and artefacts.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (set -1 for CPU).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases experiment tracking.",
    )
    return parser.parse_args()


def main() -> None:
    """End-to-end training pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ── logging & reproducibility ────────────────────────────────────────
    _setup_logging(output_dir)
    _set_seed(args.seed)

    # ── config ───────────────────────────────────────────────────────────
    cfg = _load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    epochs: int = train_cfg.get("epochs", 50)
    batch_size: int = train_cfg.get("batch_size", 32)
    lr: float = train_cfg.get("learning_rate", 1e-3)
    weight_decay: float = train_cfg.get("weight_decay", 1e-4)
    focal_gamma: float = train_cfg.get("focal_gamma", 2.0)
    patience: int = train_cfg.get("patience", 7)
    grad_clip: float = train_cfg.get("grad_clip", 1.0)

    logger.info("Training config: epochs=%d  batch=%d  lr=%.2e  wd=%.2e",
                epochs, batch_size, lr, weight_decay)

    # ── device ───────────────────────────────────────────────────────────
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ── data ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_dataset = SpeciesDataset(
        root_dir=data_dir / "train",
        config=data_cfg,
        split="train",
    )
    val_dataset = SpeciesDataset(
        root_dir=data_dir / "val",
        config=data_cfg,
        split="val",
    )

    logger.info("Train samples: %d | Val samples: %d",
                len(train_dataset), len(val_dataset))

    # Class distribution plot
    plot_class_distribution(
        labels=train_dataset.get_labels(),
        class_names=IUCN_CLASSES,
        save_path=output_dir / "class_distribution.png",
    )

    sampler = _build_stratified_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    # ── model ────────────────────────────────────────────────────────────
    model = ExtinctionRiskModel(
        num_classes=len(IUCN_CLASSES),
        **model_cfg,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s total, %s trainable",
                f"{total_params:,}", f"{trainable:,}")

    # ── loss, optimiser, scheduler ───────────────────────────────────────
    class_weights = _build_class_weights(train_dataset, device)
    criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # ── optional: resume from checkpoint ─────────────────────────────────
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_metric = ckpt.get("best_metric", 0.0)
            logger.info("Resumed from epoch %d (best_metric=%.4f)", start_epoch, best_metric)
        else:
            logger.warning("Checkpoint %s not found – training from scratch.", ckpt_path)

    # ── W&B ──────────────────────────────────────────────────────────────
    if args.wandb:
        try:
            import wandb

            wandb.init(
                project="species-extinction-predictor",
                config={**cfg, "seed": args.seed, "device": str(device)},
            )
            wandb.watch(model, log="gradients", log_freq=100)
            logger.info("Weights & Biases tracking enabled.")
        except ImportError:
            logger.warning("wandb not installed – skipping experiment tracking.")
            args.wandb = False

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        class_names=IUCN_CLASSES,
        grad_clip=grad_clip,
        patience=patience,
        start_epoch=start_epoch,
        best_metric=best_metric,
        use_wandb=args.wandb,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    # ── save final artefacts ─────────────────────────────────────────────
    # 1. Training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    # 2. Training curves
    plot_training_curves(
        history_json=str(history_path),
        save_dir=str(output_dir),
    )

    # 3. Final model checkpoint
    final_ckpt = output_dir / "final_model.pth"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "class_names": IUCN_CLASSES,
        },
        final_ckpt,
    )
    logger.info("Final model saved to %s", final_ckpt)

    # 4. Classification report (text)
    report_path = output_dir / "classification_report.txt"
    if hasattr(trainer, "last_classification_report"):
        with open(report_path, "w") as f:
            f.write(trainer.last_classification_report)
        logger.info("Classification report saved to %s", report_path)

    logger.info("✅ Training complete.")


if __name__ == "__main__":
    main()
