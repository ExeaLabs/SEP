"""
Static Visualisation Utilities
===============================
Publication-quality matplotlib/seaborn plots for training diagnostics,
class distributions, attention maps, and feature importance.

All functions follow the convention::

    plot_*(data_args, save_path, **kwargs) -> None

Dependencies:
    pip install matplotlib seaborn
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15, palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# IUCN colour palette for consistent branding
IUCN_PALETTE: Dict[str, str] = {
    "LC": "#4CAF50",
    "NT": "#CDDC39",
    "VU": "#FF9800",
    "EN": "#F44336",
    "CR": "#8B0000",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Training curves
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    history_json: str,
    save_dir: str,
    *,
    figsize: tuple = (14, 5),
) -> None:
    """Plot loss and metric curves from a training history JSON file.

    Expected JSON structure::

        {
          "train_loss": [0.8, ...],
          "val_loss": [0.9, ...],
          "train_f1": [0.3, ...],
          "val_f1": [0.4, ...],
          "learning_rate": [1e-3, ...]
        }

    Parameters
    ----------
    history_json : str
        Path to the JSON file written by the trainer.
    save_dir : str
        Directory in which to save the plot PNG.
    """
    path = Path(history_json)
    with open(path, "r") as f:
        history: Dict[str, List[float]] = json.load(f)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        logger.warning("Empty training history — skipping plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Loss ─────────────────────────────────────────────────────────────
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── F1 Score ─────────────────────────────────────────────────────────
    ax = axes[1]
    if "train_f1" in history:
        ax.plot(epochs, history["train_f1"], "o-", label="Train F1", color="#2ca02c")
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], "s--", label="Val F1", color="#d62728")
    ax.axhline(y=0.80, color="gray", linestyle=":", label="Target (0.80)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title("F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learning Rate ────────────────────────────────────────────────────
    ax = axes[2]
    if "learning_rate" in history:
        ax.plot(epochs, history["learning_rate"], "^-", color="#9467bd")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Summary", fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Training curves saved to %s", save_path)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Class distribution
# ═════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(
    labels: Union[np.ndarray, Sequence[int]],
    class_names: List[str],
    save_path: Union[str, Path],
    *,
    figsize: tuple = (8, 5),
) -> None:
    """Bar chart showing the number of samples per IUCN class.

    Parameters
    ----------
    labels : array-like of int
        Integer-encoded labels (0 … C-1).
    class_names : list of str
        Human-readable class names (e.g. ["LC", "NT", …]).
    save_path : str or Path
        Output file path.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=len(class_names))

    colours = [IUCN_PALETTE.get(c, "#999") for c in class_names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(class_names, counts, color=colours, edgecolor="white", linewidth=1.2)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("IUCN Category")
    ax.set_ylabel("Number of Species")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Class distribution plot saved to %s", save_path)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Attention heatmap overlay
# ═════════════════════════════════════════════════════════════════════════════

def plot_attention_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    save_path: Union[str, Path],
    *,
    alpha: float = 0.5,
    cmap: str = "jet",
    figsize: tuple = (8, 8),
) -> None:
    """Overlay a spatial attention map on a satellite image.

    Parameters
    ----------
    image : np.ndarray
        RGB image array, shape (H, W, 3), values in [0, 255] or [0, 1].
    attention_map : np.ndarray
        2-D attention weights, shape (h, w). Will be resized to match image.
    save_path : str or Path
        Output file path.
    alpha : float
        Opacity of the attention overlay.
    cmap : str
        Matplotlib colourmap for the attention heatmap.
    """
    from scipy.ndimage import zoom as ndimage_zoom

    # Normalise image to [0, 1]
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    # Resize attention to match image dimensions
    h_img, w_img = img.shape[:2]
    h_attn, w_attn = attention_map.shape
    if (h_attn, w_attn) != (h_img, w_img):
        zoom_h = h_img / h_attn
        zoom_w = w_img / w_attn
        attention_map = ndimage_zoom(attention_map, (zoom_h, zoom_w), order=1)

    # Normalise attention to [0, 1]
    attn = attention_map.astype(np.float64)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.8, figsize[1]))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")

    # Attention map
    im = axes[1].imshow(attn, cmap=cmap)
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attn, cmap=cmap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle("Spatial Attention Analysis", fontsize=15)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Attention heatmap saved to %s", save_path)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Temporal attention
# ═════════════════════════════════════════════════════════════════════════════

def plot_temporal_attention(
    years: Sequence[int],
    attention_weights: np.ndarray,
    save_path: Union[str, Path],
    *,
    figsize: tuple = (10, 4),
) -> None:
    """Visualise which years the model attends to most strongly.

    Parameters
    ----------
    years : sequence of int
        Calendar years corresponding to temporal tokens.
    attention_weights : np.ndarray
        1-D array of attention weights aligned with ``years``.
    save_path : str or Path
        Output file path.
    """
    weights = np.asarray(attention_weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [str(y) for y in years],
        weights,
        color=plt.cm.YlOrRd(weights / (weights.max() + 1e-8)),
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Temporal Attention — Which Years Matter?")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Temporal attention plot saved to %s", save_path)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Feature importance / attribution
# ═════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model: Any,
    sample_batch: Dict[str, Any],
    save_path: Union[str, Path],
    *,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    figsize: tuple = (10, 6),
) -> None:
    """Compute and plot feature importance via gradient-based attribution.

    Uses the integrated-gradients-lite approach: compute ∂loss/∂input for a
    sample batch, average absolute gradients, and rank features.

    Parameters
    ----------
    model : nn.Module
        Trained model (must be on the correct device).
    sample_batch : dict
        A single batch dict from the DataLoader.
    save_path : str or Path
        Output file path.
    feature_names : list of str, optional
        Human-readable feature names. Falls back to ``Feature_0``, etc.
    top_k : int
        Number of top features to display.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device

    # Prepare input with gradient tracking
    inputs = {}
    grad_target = None
    for k, v in sample_batch.items():
        if k in ("label", "species_name"):
            continue
        if isinstance(v, torch.Tensor):
            v = v.to(device).float().requires_grad_(True)
            if grad_target is None:
                grad_target = v
        inputs[k] = v

    if grad_target is None:
        logger.warning("No tensor input found in batch – cannot compute attribution.")
        return

    # Forward pass
    outputs = model(**inputs)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs

    # Sum logits (class-agnostic importance)
    logits.sum().backward()

    if grad_target.grad is None:
        logger.warning("Gradients are None — model may not back-propagate to input.")
        return

    # Average absolute gradient across the batch → feature importance
    importance = grad_target.grad.abs().mean(dim=0).detach().cpu().numpy()

    # Flatten if multi-dimensional
    if importance.ndim > 1:
        importance = importance.mean(axis=tuple(range(1, importance.ndim)))

    n_features = len(importance)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    feature_names = feature_names[:n_features]

    # Sort and select top-k
    sorted_idx = np.argsort(importance)[::-1][:top_k]
    top_names = [feature_names[i] for i in sorted_idx]
    top_values = importance[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
    ax.barh(
        range(len(top_names)),
        top_values[::-1],
        color=colours,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Mean |Gradient|")
    ax.set_title(f"Top-{top_k} Feature Importance (Gradient Attribution)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path)
