#!/usr/bin/env python3
"""
Species Extinction Risk Predictor — Evaluation Script
======================================================
Compute comprehensive metrics, generate diagnostic plots, and validate
model attention against known deforestation hotspots.

Usage:
    python evaluate.py \\
        --model-path outputs/best_model.pth \\
        --data-dir data/ \\
        --output-dir evaluation/ \\
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.model import ExtinctionRiskModel
from src.data.dataset import SpeciesDataset

# ── constants ────────────────────────────────────────────────────────────────
IUCN_CLASSES: List[str] = ["LC", "NT", "VU", "EN", "CR"]

# Publication-quality plotting defaults
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

logger = logging.getLogger("evaluate")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"evaluate_{datetime.now():%Y%m%d_%H%M%S}.log"

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


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_model(
    model_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> ExtinctionRiskModel:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = config.get("model", {})
    model = ExtinctionRiskModel(num_classes=len(IUCN_CLASSES), **model_cfg)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation logic
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_predictions(
    model: ExtinctionRiskModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """Run full-pass inference and collect predictions, labels, probabilities.

    Returns
    -------
    all_labels : (N,)
    all_preds  : (N,)
    all_probs  : (N, C)
    attention_maps : list of attention weight arrays, or None
    """
    labels_list: List[np.ndarray] = []
    preds_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    attention_maps: List[np.ndarray] = []

    for batch in dataloader:
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items() if k not in ("species_name", "label")}
        targets = batch["label"].numpy()

        outputs = model(**inputs)

        # Handle models that return (logits, attention_weights)
        if isinstance(outputs, tuple):
            logits, attn = outputs
            if attn is not None:
                attention_maps.append(attn.cpu().numpy())
        else:
            logits = outputs

        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

        labels_list.append(targets)
        preds_list.append(preds)
        probs_list.append(probs)

    all_labels = np.concatenate(labels_list)
    all_preds = np.concatenate(preds_list)
    all_probs = np.concatenate(probs_list)
    attn_out = attention_maps if len(attention_maps) > 0 else None
    return all_labels, all_preds, all_probs, attn_out


def _compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Any]:
    """Compute a comprehensive set of evaluation metrics."""
    # Overall
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    # Per-class
    precision, recall, f1_per, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(IUCN_CLASSES))), zero_division=0,
    )

    # AUC-ROC (one-vs-rest)
    try:
        auc_roc = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
        )
    except ValueError:
        auc_roc = float("nan")

    per_class: Dict[str, Dict[str, float]] = {}
    for i, cls in enumerate(IUCN_CLASSES):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "auc_roc": float(auc_roc),
        "per_class": per_class,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
) -> None:
    """Plot and save a normalised confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(IUCN_CLASSES))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=IUCN_CLASSES, yticklabels=IUCN_CLASSES,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalised
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=IUCN_CLASSES, yticklabels=IUCN_CLASSES,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalised)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def _plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    """Plot one-vs-rest ROC curves for each IUCN class."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colours = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728", "#8b0000"]

    for i, (cls, colour) in enumerate(zip(IUCN_CLASSES, colours)):
        binary = (labels == i).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)


def _validate_attention_maps(
    attention_maps: List[np.ndarray],
    output_dir: Path,
) -> Dict[str, Any]:
    """Validate model attention against known deforestation hotspots.

    Uses a predefined set of known deforestation regions (bounding boxes) to
    check whether the spatial attention weights concentrate in these areas.
    """
    # Known deforestation hotspots (approximate bounding boxes)
    HOTSPOTS = {
        "Amazon Basin":     {"lat": (-15, 5),   "lon": (-75, -45)},
        "Congo Basin":      {"lat": (-5, 5),    "lon": (15, 30)},
        "Southeast Asia":   {"lat": (-10, 20),  "lon": (95, 140)},
        "Madagascar":       {"lat": (-26, -12), "lon": (43, 50)},
        "Atlantic Forest":  {"lat": (-30, -5),  "lon": (-55, -35)},
    }

    results: Dict[str, Any] = {}
    logger.info("Validating attention maps against %d deforestation hotspots …",
                len(HOTSPOTS))

    for name, bbox in HOTSPOTS.items():
        # In a full implementation, we would extract attention weights
        # that fall within the bbox and compute the fraction of total
        # attention mass concentrated there.
        results[name] = {
            "bbox": bbox,
            "attention_mass": "N/A — requires georeferenced attention maps",
            "status": "pending",
        }

    report_path = output_dir / "attention_validation.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Attention validation report saved to %s", report_path)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Species Extinction Risk Predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/",
        help="Root data directory (must contain a 'test/' split).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (-1 for CPU).",
    )
    parser.add_argument(
        "--validate-attention", action="store_true",
        help="Validate attention maps against known deforestation hotspots.",
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

    # ── config, model, data ──────────────────────────────────────────────
    cfg = _load_config(args.config)
    model = _load_model(args.model_path, cfg, device)

    test_dataset = SpeciesDataset(
        root_dir=Path(args.data_dir) / "test",
        config=cfg.get("data", {}),
        split="test",
    )
    logger.info("Test samples: %d", len(test_dataset))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, __import__("os").cpu_count() or 1),
    )

    # ── inference ────────────────────────────────────────────────────────
    labels, preds, probs, attn_maps = _collect_predictions(model, test_loader, device)

    # ── metrics ──────────────────────────────────────────────────────────
    metrics = _compute_metrics(labels, preds, probs)

    logger.info("─── Evaluation Results ───")
    logger.info("  Accuracy   : %.4f", metrics["accuracy"])
    logger.info("  Macro F1   : %.4f", metrics["macro_f1"])
    logger.info("  Weighted F1: %.4f", metrics["weighted_f1"])
    logger.info("  AUC-ROC    : %.4f", metrics["auc_roc"])
    for cls, vals in metrics["per_class"].items():
        logger.info("  %s — P=%.3f  R=%.3f  F1=%.3f  n=%d",
                     cls, vals["precision"], vals["recall"],
                     vals["f1"], vals["support"])
    logger.info("──────────────────────────")

    # Save JSON metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ── classification report (text) ─────────────────────────────────────
    report = classification_report(
        labels, preds,
        target_names=IUCN_CLASSES,
        zero_division=0,
    )
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Classification report:\n%s", report)

    # ── plots ────────────────────────────────────────────────────────────
    _plot_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
    _plot_roc_curves(labels, probs, output_dir / "roc_curves.png")

    # ── attention validation (optional) ──────────────────────────────────
    if args.validate_attention and attn_maps is not None:
        _validate_attention_maps(attn_maps, output_dir)

    # ── target check ─────────────────────────────────────────────────────
    TARGET_F1 = 0.80
    TARGET_AUC = 0.92
    f1_ok = metrics["macro_f1"] >= TARGET_F1
    auc_ok = metrics["auc_roc"] >= TARGET_AUC

    logger.info("Target check — Macro F1 ≥ %.2f: %s", TARGET_F1, "✅ PASS" if f1_ok else "❌ FAIL")
    logger.info("Target check — AUC-ROC ≥ %.2f: %s", TARGET_AUC, "✅ PASS" if auc_ok else "❌ FAIL")

    logger.info("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
