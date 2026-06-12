"""Evaluation Metrics and Visualisation for Extinction Risk Classification.

Provides functions for computing classification metrics (macro F1, AUC-ROC,
per-class precision/recall/F1) and generating publication-quality plots
(confusion matrices, ROC curves) for the 5-class IUCN risk prediction task.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Default IUCN class names
DEFAULT_CLASS_NAMES: list[str] = [
    "Least Concern",
    "Near Threatened",
    "Vulnerable",
    "Endangered",
    "Critically Endangered",
]


def _to_numpy(x: Union[np.ndarray, "torch.Tensor", list]) -> np.ndarray:
    """Convert input to a NumPy array, detaching from GPU if necessary."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ======================================================================
# Scalar Metrics
# ======================================================================


def compute_macro_f1(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute macro-averaged F1 score across all classes.

    Args:
        y_true: Ground-truth class labels, shape ``(N,)``.
        y_pred: Predicted class labels, shape ``(N,)``.

    Returns:
        Macro F1 score in ``[0, 1]``.

    Raises:
        ValueError: If inputs have different lengths.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    score: float = f1_score(y_true, y_pred, average="macro", zero_division=0)
    logger.debug("Macro F1: %.4f", score)
    return score


def compute_auc_roc(
    y_true: Union[np.ndarray, list],
    y_probs: Union[np.ndarray, list],
) -> float:
    """Compute one-vs-rest macro-averaged AUC-ROC.

    Args:
        y_true: Ground-truth class labels, shape ``(N,)``.
        y_probs: Predicted class probabilities, shape ``(N, C)``.

    Returns:
        Macro-averaged AUC-ROC score.

    Raises:
        ValueError: If fewer than 2 classes are present in ``y_true``.
    """
    y_true = _to_numpy(y_true)
    y_probs = _to_numpy(y_probs)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            "AUC-ROC requires ≥ 2 classes in y_true; found %d. Returning 0.0.",
            len(unique_classes),
        )
        return 0.0

    try:
        score: float = roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro"
        )
    except ValueError as exc:
        logger.warning("AUC-ROC computation failed: %s. Returning 0.0.", exc)
        return 0.0

    logger.debug("Macro AUC-ROC: %.4f", score)
    return score


# ======================================================================
# Per-Class Metrics
# ======================================================================


def compute_per_class_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, and F1 for each class.

    Args:
        y_true: Ground-truth labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        class_names: Human-readable names for each class.  Defaults to
            the standard IUCN categories.

    Returns:
        Dictionary mapping class name to a dict of
        ``{"precision", "recall", "f1", "support"}``.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    results: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        results[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    logger.debug("Per-class metrics: %s", results)
    return results


# ======================================================================
# Visualisation
# ======================================================================


def plot_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    class_names: Optional[list[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "YlOrRd",
    normalize: bool = True,
) -> plt.Figure:
    """Generate a publication-quality confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Class display names.
        save_path: If provided, save the figure to this path (PNG, 300 dpi).
        figsize: Figure size in inches.
        cmap: Seaborn/Matplotlib colourmap name.
        normalize: If ``True``, normalise rows to show percentages.

    Returns:
        Matplotlib ``Figure`` object.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums * 100
        fmt = ".1f"
        title_suffix = " (Normalised %)"
    else:
        cm_display = cm.astype(float)
        fmt = ".0f"
        title_suffix = " (Counts)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Percentage (%)" if normalize else "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Species Extinction Risk — Confusion Matrix{title_suffix}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_roc_curves(
    y_true: Union[np.ndarray, list],
    y_probs: Union[np.ndarray, list],
    class_names: Optional[list[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true: Ground-truth labels, shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N, C)``.
        class_names: Class display names.
        save_path: If provided, save the figure (PNG, 300 dpi).
        figsize: Figure size in inches.

    Returns:
        Matplotlib ``Figure`` object.
    """
    y_true = _to_numpy(y_true)
    y_probs = _to_numpy(y_probs)
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    num_classes = len(class_names)
    # Binarise labels for one-vs-rest
    y_true_bin = np.eye(num_classes)[y_true.astype(int)]  # (N, C)

    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, (name, colour) in enumerate(zip(class_names, colours)):
        if y_true_bin[:, i].sum() == 0:
            logger.warning("No positive samples for class '%s'; skipping ROC.", name)
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colour,
            lw=2,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    # Chance line
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    ax.set_title(
        "Species Extinction Risk — One-vs-Rest ROC Curves",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)

    return fig


# ======================================================================
# Full Report
# ======================================================================


def generate_classification_report(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_probs: Union[np.ndarray, list],
    class_names: Optional[list[str]] = None,
    output_dir: Union[str, Path] = "reports",
) -> dict:
    """Generate a comprehensive classification report with metrics and plots.

    Creates:
        - ``metrics.json``: All scalar and per-class metrics.
        - ``confusion_matrix.png``: Normalised confusion matrix heatmap.
        - ``roc_curves.png``: Per-class ROC curves.
        - ``classification_report.txt``: Sklearn text report.

    Args:
        y_true: Ground-truth labels, shape ``(N,)``.
        y_pred: Predicted labels, shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N, C)``.
        class_names: Class display names.
        output_dir: Directory to write report artefacts.

    Returns:
        Dictionary containing all computed metrics.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    y_probs = _to_numpy(y_probs)

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    macro_f1 = compute_macro_f1(y_true, y_pred)
    auc_roc = compute_auc_roc(y_true, y_probs)
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)

    report_dict = {
        "macro_f1": macro_f1,
        "auc_roc": auc_roc,
        "num_samples": int(len(y_true)),
        "num_classes": len(class_names),
        "per_class": per_class,
    }

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ------------------------------------------------------------------
    # Save sklearn text report
    # ------------------------------------------------------------------
    text_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    report_txt_path = output_dir / "classification_report.txt"
    with open(report_txt_path, "w") as f:
        f.write("Species Extinction Risk — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(text_report)
        f.write(f"\nMacro F1:  {macro_f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc_roc:.4f}\n")
    logger.info("Text report saved to %s", report_txt_path)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
    )

    plot_roc_curves(
        y_true,
        y_probs,
        class_names=class_names,
        save_path=output_dir / "roc_curves.png",
    )

    # Close all figures to free memory
    plt.close("all")

    logger.info("Full classification report generated in %s", output_dir)
    return report_dict
