"""Training engine for the Species Extinction Risk Predictor.

Follows the EPGNN pattern: load dataset from CSV + HDF5, split train/val,
run epochs with tqdm, save checkpoint to project root.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from data.dataset import SpeciesExtinctionDataset, IUCN_CLASSES
from models.model import ExtinctionRiskModel
from losses.focal_loss import FocalLoss


def train_model(
    epochs: int = 5,
    batch_size: int = 16,
    metadata_path: str = "metadata_clean.csv",
    hdf5_path: str = "modalities.hdf5",
    model_path: str = "extinction_model.pth",
    learning_rate: float = 1e-4,
    freeze_backbone: bool = True,
) -> None:
    """Train the multimodal extinction risk classifier."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
        print("Enabled cuDNN autotuning for high performance.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        dataset = SpeciesExtinctionDataset(
            metadata_path=metadata_path,
            hdf5_path=hdf5_path,
        )
    except FileNotFoundError as exc:
        print(f"Dataset not found: {exc}")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    train_labels = np.array([dataset.labels[i] for i in train_dataset.indices])
    counts = np.bincount(train_labels, minlength=len(IUCN_CLASSES))
    weights = 1.0 / (counts + 1e-6)
    sample_weights = weights[train_labels]
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = ExtinctionRiskModel(freeze_backbone=freeze_backbone).to(device)

    labels = np.array(dataset.get_labels())
    counts = np.bincount(labels, minlength=len(IUCN_CLASSES)).astype(np.float64)
    alpha = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    alpha = alpha / alpha.sum() * len(IUCN_CLASSES)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for satellite, climate, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        ):
            satellite = satellite.to(device, non_blocking=True)
            climate = climate.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _, _ = model(satellite, climate)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for satellite, climate, targets in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False
            ):
                satellite = satellite.to(device, non_blocking=True)
                climate = climate.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, _, _ = model(satellite, climate)
                    loss = criterion(logits, targets)

                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  ★ Saved best model to {model_path}")

    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")
