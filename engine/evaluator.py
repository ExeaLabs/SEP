"""Evaluation engine for the Species Extinction Risk Predictor."""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SpeciesExtinctionDataset, IUCN_CLASSES
from models.model import ExtinctionRiskModel


def evaluate_model(
    batch_size: int = 16,
    metadata_path: str = "metadata_clean.csv",
    hdf5_path: str = "modalities.hdf5",
    model_path: str = "extinction_model.pth",
    freeze_backbone: bool = True,
) -> None:
    """Evaluate a trained extinction risk model on the full dataset."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        dataset = SpeciesExtinctionDataset(
            metadata_path=metadata_path,
            hdf5_path=hdf5_path,
        )
    except FileNotFoundError as exc:
        print(f"Dataset not found: {exc}")
        return

    num_workers = min(4, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = ExtinctionRiskModel(freeze_backbone=freeze_backbone).to(device)
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    except Exception as exc:
        print(f"Could not load model weights from {model_path}: {exc}")
        return

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for satellite, climate, targets in tqdm(loader, desc="Evaluating"):
            satellite = satellite.to(device, non_blocking=True)
            climate = climate.to(device, non_blocking=True)

            logits, _, _ = model(satellite, climate)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(targets.numpy().tolist())
            all_probs.append(probs)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:       {acc * 100:.2f}%")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Samples:        {len(all_labels)}")
    print("\nPer-class report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=IUCN_CLASSES,
        labels=list(range(len(IUCN_CLASSES))),
        zero_division=0,
    ))
    print("--------------------------")
