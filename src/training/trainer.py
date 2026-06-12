"""Training Engine for the Species Extinction Risk Predictor.

Full-featured training loop with mixed-precision training, gradient
clipping, cosine annealing with warmup, early stopping, checkpoint
management, and optional Weights & Biases integration.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    from tqdm import tqdm

from src.training.metrics import compute_auc_roc, compute_macro_f1

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class TrainerConfig:
    """Training hyper-parameters and settings.

    Attributes:
        epochs: Maximum number of training epochs.
        lr: Peak learning rate for AdamW.
        weight_decay: L2 regularisation coefficient.
        warmup_epochs: Number of linear warmup epochs for the LR scheduler.
        grad_clip_norm: Maximum gradient norm for clipping (0 = disabled).
        use_amp: Enable automatic mixed-precision training.
        early_stopping_patience: Number of epochs without val improvement
            before stopping.  Set to 0 to disable.
        checkpoint_dir: Directory for saving model checkpoints.
        save_every_n_epochs: Save a periodic checkpoint every N epochs.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_run_name: Optional W&B run name.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        num_classes: Number of output classes.
    """

    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    use_wandb: bool = False
    wandb_project: str = "species-extinction-predictor"
    wandb_run_name: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainerConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        import dataclasses

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ======================================================================
# Learning-rate schedule: cosine annealing with linear warmup
# ======================================================================


def _cosine_warmup_lambda(
    current_epoch: int,
    warmup_epochs: int,
    total_epochs: int,
) -> float:
    """LR multiplier for cosine schedule with linear warmup."""
    if current_epoch < warmup_epochs:
        return (current_epoch + 1) / max(warmup_epochs, 1)
    # Cosine decay from 1 → 0 over remaining epochs
    progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


# ======================================================================
# Trainer
# ======================================================================


class Trainer:
    """Training engine for the MultimodalFusionNetwork.

    Handles the full training lifecycle: epoch loops, validation,
    metric tracking, LR scheduling, checkpointing, and early stopping.

    Args:
        model: The neural network to train.
        criterion: Loss function (e.g., ``FocalLoss``).
        config: ``TrainerConfig`` or a dict of config values.

    Example::

        trainer = Trainer(model, criterion, config)
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: Union[TrainerConfig, dict[str, Any]],
    ) -> None:
        if isinstance(config, dict):
            config = TrainerConfig.from_dict(config)
        self.config = config

        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        # Optimiser
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: _cosine_warmup_lambda(
                epoch, config.warmup_epochs, config.epochs
            ),
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            device=config.device, enabled=config.use_amp
        )
        self.use_amp = config.use_amp

        # Training state
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "val_auc_roc": [],
            "lr": [],
        }
        self.best_val_f1: float = 0.0
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        self._wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        logger.info("Trainer initialised on device=%s", self.device)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialise Weights & Biases run."""
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "epochs": self.config.epochs,
                    "lr": self.config.lr,
                    "weight_decay": self.config.weight_decay,
                    "warmup_epochs": self.config.warmup_epochs,
                    "use_amp": self.config.use_amp,
                    "grad_clip_norm": self.config.grad_clip_norm,
                },
            )
            logger.info("W&B run initialised: %s", self._wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging.")
            self.config.use_wandb = False
        except Exception as exc:
            logger.warning("Failed to initialise W&B: %s", exc)
            self.config.use_wandb = False

    def _log_wandb(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B if enabled."""
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as exc:
                logger.debug("W&B logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Single epoch: train
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.  Each batch should yield
                a dict or tuple of ``(satellite_images, climate_series,
                targets)`` or ``{"satellite": ..., "climate": ...,
                "target": ...}``.
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary with ``"loss"`` and ``"macro_f1"`` for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]",
            leave=False,
        )

        for batch in pbar:
            satellite, climate, targets = self._unpack_batch(batch)
            satellite = satellite.to(self.device, non_blocking=True)
            climate = climate.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            amp_device = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=self.use_amp):
                logits, _, _ = self.model(satellite, climate)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            running_loss += loss.item()
            num_batches += 1
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(num_batches, 1)
        all_preds_arr = np.concatenate(all_preds)
        all_targets_arr = np.concatenate(all_targets)
        macro_f1 = compute_macro_f1(all_targets_arr, all_preds_arr)

        return {"loss": avg_loss, "macro_f1": macro_f1}

    # ------------------------------------------------------------------
    # Single epoch: validate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one validation epoch.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary with ``"loss"``, ``"macro_f1"``, and ``"auc_roc"``.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        num_batches = 0

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs} [Val]",
            leave=False,
        )

        for batch in pbar:
            satellite, climate, targets = self._unpack_batch(batch)
            satellite = satellite.to(self.device, non_blocking=True)
            climate = climate.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            amp_device = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=self.use_amp):
                logits, _, _ = self.model(satellite, climate)
                loss = self.criterion(logits, targets)

            running_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs)

        avg_loss = running_loss / max(num_batches, 1)
        all_preds_arr = np.concatenate(all_preds)
        all_targets_arr = np.concatenate(all_targets)
        all_probs_arr = np.concatenate(all_probs)

        macro_f1 = compute_macro_f1(all_targets_arr, all_preds_arr)
        auc_roc = compute_auc_roc(all_targets_arr, all_probs_arr)

        return {"loss": avg_loss, "macro_f1": macro_f1, "auc_roc": auc_roc}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, list[float]]:
        """Run the full training loop with validation.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Training history dictionary with per-epoch metrics.
        """
        logger.info(
            "Starting training for %d epochs (device=%s, AMP=%s)",
            self.config.epochs,
            self.device,
            self.use_amp,
        )
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Step LR scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_macro_f1"].append(train_metrics["macro_f1"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_auc_roc"].append(val_metrics["auc_roc"])
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Log
            log_msg = (
                f"Epoch {epoch + 1}/{self.config.epochs} "
                f"({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train F1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1: {val_metrics['macro_f1']:.4f}, "
                f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            logger.info(log_msg)
            print(log_msg)

            # W&B logging
            self._log_wandb(
                {
                    "train/loss": train_metrics["loss"],
                    "train/macro_f1": train_metrics["macro_f1"],
                    "val/loss": val_metrics["loss"],
                    "val/macro_f1": val_metrics["macro_f1"],
                    "val/auc_roc": val_metrics["auc_roc"],
                    "lr": current_lr,
                },
                step=epoch,
            )

            # ----------------------------------------------------------
            # Checkpointing
            # ----------------------------------------------------------
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(
                    "★ New best model at epoch %d (Val F1: %.4f)",
                    epoch + 1,
                    self.best_val_f1,
                )
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (
                self.config.save_every_n_epochs > 0
                and (epoch + 1) % self.config.save_every_n_epochs == 0
            ):
                self._save_checkpoint(epoch, is_best=False)

            # ----------------------------------------------------------
            # Early stopping
            # ----------------------------------------------------------
            if (
                self.config.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered after %d epochs without "
                    "improvement (best epoch: %d, best F1: %.4f)",
                    self.config.early_stopping_patience,
                    self.best_epoch + 1,
                    self.best_val_f1,
                )
                print(
                    f"\n⚠ Early stopping at epoch {epoch + 1} "
                    f"(best: epoch {self.best_epoch + 1}, F1={self.best_val_f1:.4f})"
                )
                break

        total_time = time.time() - start_time
        logger.info(
            "Training complete in %.1f minutes. Best Val F1: %.4f at epoch %d.",
            total_time / 60,
            self.best_val_f1,
            self.best_epoch + 1,
        )

        # Save history
        self._save_history()

        # Finalise W&B
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        return self.history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack a batch into (satellite, climate, targets).

        Supports:
            - Tuple/list of ``(satellite, climate, targets)``
            - Dict with keys ``"satellite"``, ``"climate"``, ``"target"``
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) != 3:
                raise ValueError(
                    f"Expected batch tuple of length 3, got {len(batch)}"
                )
            return batch[0], batch[1], batch[2]
        elif isinstance(batch, dict):
            return batch["satellite"], batch["climate"], batch["target"]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: If ``True``, save as ``best_model.pt``.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_f1": self.best_val_f1,
            "config": {
                "epochs": self.config.epochs,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "warmup_epochs": self.config.warmup_epochs,
                "grad_clip_norm": self.config.grad_clip_norm,
                "use_amp": self.config.use_amp,
                "num_classes": self.config.num_classes,
            },
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

    def _save_history(self) -> None:
        """Save training history as JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", history_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.

        Returns:
            The epoch number the checkpoint was saved at.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)

        epoch = checkpoint["epoch"]
        logger.info(
            "Checkpoint loaded from %s (epoch %d, best F1: %.4f)",
            checkpoint_path,
            epoch + 1,
            self.best_val_f1,
        )
        return epoch
