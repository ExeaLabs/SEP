"""Multimodal Fusion Network for Species Extinction Risk Prediction.

Combines satellite imagery features (SatelliteEncoder) and climate
time-series features (ClimateEncoder) via concatenation-based fusion
with learnable projection layers. Outputs IUCN extinction risk class
logits along with interpretability artefacts (spatial attention maps
and temporal attention weights).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from models.satellite_encoder import SatelliteEncoder
from models.climate_encoder import ClimateEncoder

logger = logging.getLogger(__name__)

# IUCN Red List categories
IUCN_CLASSES: list[str] = [
    "Least Concern",
    "Near Threatened",
    "Vulnerable",
    "Endangered",
    "Critically Endangered",
]


class MultimodalFusionNetwork(nn.Module):
    """Multimodal fusion network combining satellite and climate encoders.

    Architecture overview::

        satellite_image ──► SatelliteEncoder ──► sat_features (feature_dim)
                                                       │
        climate_series  ──► ClimateEncoder  ──► clim_features (climate_output_dim)
                                                       │
                            concat(sat, clim) ─────────┘
                                    │
                            FC(1024) + BN + ReLU + Dropout(0.5)
                                    │
                            FC(512) + BN + ReLU + Dropout(0.3)
                                    │
                            FC(num_classes)
                                    │
                                 logits

    Args:
        satellite_feature_dim: Output dimension of the satellite encoder.
        climate_input_dim: Number of climate variables per time step.
        climate_hidden_dim: LSTM hidden size in the climate encoder.
        climate_num_layers: Number of LSTM layers.
        climate_dropout: Dropout in the climate encoder.
        num_classes: Number of IUCN risk categories (default 5).
        pretrained_backbone: Use ImageNet-pretrained ResNet-50.
        freeze_backbone: Freeze ResNet backbone parameters.

    Example::

        model = MultimodalFusionNetwork()
        sat = torch.randn(4, 3, 224, 224)
        clim = torch.randn(4, 30, 12)
        logits, sat_attn, temp_attn = model(sat, clim)
        # logits.shape    -> (4, 5)
        # sat_attn.shape  -> (4, 1, 7, 7)
        # temp_attn.shape -> (4, 30)
    """

    def __init__(
        self,
        satellite_feature_dim: int = 512,
        climate_input_dim: int = 12,
        climate_hidden_dim: int = 256,
        climate_num_layers: int = 2,
        climate_dropout: float = 0.3,
        num_classes: int = 5,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Sub-encoders
        # ------------------------------------------------------------------
        self.satellite_encoder = SatelliteEncoder(
            feature_dim=satellite_feature_dim,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
        )

        self.climate_encoder = ClimateEncoder(
            input_dim=climate_input_dim,
            hidden_dim=climate_hidden_dim,
            num_layers=climate_num_layers,
            dropout=climate_dropout,
        )

        # ------------------------------------------------------------------
        # Fusion head
        # ------------------------------------------------------------------
        climate_output_dim = climate_hidden_dim * 2  # bidirectional
        fusion_input_dim = satellite_feature_dim + climate_output_dim

        self.fusion_head = nn.Sequential(
            # Layer 1: -> 1024
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # Layer 2: -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            # Classification layer
            nn.Linear(512, num_classes),
        )

        self._init_fusion_weights()

        logger.info(
            "MultimodalFusionNetwork initialised: sat_dim=%d, clim_dim=%d, "
            "fusion_input=%d, num_classes=%d",
            satellite_feature_dim,
            climate_output_dim,
            fusion_input_dim,
            num_classes,
        )

    def _init_fusion_weights(self) -> None:
        """Kaiming-initialise the fusion head linear layers."""
        for module in self.fusion_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MultimodalFusionNetwork":
        """Instantiate the network from a configuration dictionary.

        The dictionary keys should match the constructor parameter names.
        Unknown keys are silently ignored.

        Args:
            config: Configuration mapping.  Example::

                {
                    "satellite_feature_dim": 512,
                    "climate_input_dim": 12,
                    "climate_hidden_dim": 256,
                    "num_classes": 5,
                    "pretrained_backbone": True,
                    "freeze_backbone": False,
                }

        Returns:
            A configured ``MultimodalFusionNetwork`` instance.
        """
        import inspect

        valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        logger.info("Creating MultimodalFusionNetwork from config: %s", filtered)
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_trainable_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_params(self) -> int:
        """Total number of parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        satellite_image: torch.Tensor,
        climate_series: torch.Tensor,
        climate_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full multimodal forward pass.

        Args:
            satellite_image: Satellite image batch, ``(B, 3, H, W)``.
            climate_series: Climate time-series batch, ``(B, T, C_in)``.
            climate_mask: Optional boolean mask for padded time steps,
                ``(B, T)``.

        Returns:
            Tuple of:
                - logits: Raw class scores, ``(B, num_classes)``.
                - satellite_attention_map: Spatial attention from the
                  satellite encoder, ``(B, 1, H', W')``.
                - temporal_attention_weights: Temporal attention from the
                  climate encoder, ``(B, T)``.
        """
        # Encode each modality
        sat_features, sat_attention = self.satellite_encoder(satellite_image)
        clim_features, temp_attention = self.climate_encoder(
            climate_series, mask=climate_mask
        )

        # Concatenation-based fusion
        fused = torch.cat([sat_features, clim_features], dim=-1)

        # Classification
        logits = self.fusion_head(fused)

        return logits, sat_attention, temp_attention
