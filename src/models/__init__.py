"""Species Extinction Risk Predictor - Model Architecture Components.

This package provides the neural network modules for predicting species
extinction risk from multimodal environmental data (satellite imagery
and climate time-series).

IUCN Extinction Risk Categories (5 classes):
    0: Least Concern (LC)
    1: Near Threatened (NT)
    2: Vulnerable (VU)
    3: Endangered (EN)
    4: Critically Endangered (CR)
"""

from src.models.satellite_encoder import SatelliteEncoder, SpatialAttention
from src.models.climate_encoder import ClimateEncoder, TemporalAttention
from src.models.fusion import MultimodalFusionNetwork

__all__ = [
    "SatelliteEncoder",
    "SpatialAttention",
    "ClimateEncoder",
    "TemporalAttention",
    "MultimodalFusionNetwork",
]
