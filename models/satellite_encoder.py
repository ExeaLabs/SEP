"""Satellite Imagery Encoder with Spatial Attention.

CNN encoder built on a pretrained ResNet-50 backbone for extracting
habitat-relevant features from satellite imagery. Includes a spatial
attention module that highlights deforestation-relevant regions, enabling
interpretable predictions via attention maps and Grad-CAM compatibility.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


class SpatialAttention(nn.Module):
    """Spatial attention module that learns to highlight relevant image regions.

    Applies a lightweight convolutional attention mechanism on top of CNN
    feature maps to weight spatial locations by ecological relevance
    (e.g., deforestation hotspots, habitat fragmentation boundaries).

    Architecture:
        conv1x1(in_channels -> mid_channels) -> ReLU ->
        conv1x1(mid_channels -> 1) -> Sigmoid

    Args:
        in_channels: Number of input feature map channels.
        mid_channels: Number of channels in the hidden attention layer.
            Defaults to ``in_channels // 4``.
    """

    def __init__(self, in_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = max(in_channels // 4, 1)

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute spatially-attended features.

        Args:
            feature_map: CNN feature map of shape ``(B, C, H, W)``.

        Returns:
            Tuple of:
                - attended_features: Element-wise product of the input
                  feature map and the broadcast attention weights,
                  shape ``(B, C, H, W)``.
                - attention_map: Spatial attention weights in ``[0, 1]``,
                  shape ``(B, 1, H, W)``.
        """
        attention_map: torch.Tensor = self.attention(feature_map)  # (B, 1, H, W)
        attended_features = feature_map * attention_map  # broadcast over C
        return attended_features, attention_map


class SatelliteEncoder(nn.Module):
    """CNN encoder for satellite imagery using a pretrained ResNet-50 backbone.

    Extracts spatial features from satellite images and applies a learned
    spatial attention mechanism to emphasise ecologically relevant regions.
    The final FC layer of ResNet-50 is removed and replaced with an adaptive
    average pool followed by a linear projection to ``feature_dim``.

    Args:
        feature_dim: Dimensionality of the output feature vector.
        pretrained: Whether to load ImageNet-pretrained weights.
        freeze_backbone: If ``True``, all backbone (ResNet) parameters are
            frozen so only the attention module and projection head are
            trained.  Useful for transfer-learning warm-up.
        attention_mid_channels: Hidden channels in the spatial attention
            module.  Defaults to ``512`` (= 2048 // 4).

    Example::

        encoder = SatelliteEncoder(feature_dim=512, freeze_backbone=True)
        img = torch.randn(4, 3, 224, 224)
        features, attn_map = encoder(img)
        # features.shape  -> (4, 512)
        # attn_map.shape  -> (4, 1, 7, 7)
    """

    # ResNet-50 final conv block output channels
    _BACKBONE_OUT_CHANNELS: int = 2048

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        attention_mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.feature_dim = feature_dim

        # ------------------------------------------------------------------
        # Backbone: ResNet-50 without the final avgpool + FC
        # ------------------------------------------------------------------
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Keep everything up to (and including) layer4
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        if freeze_backbone:
            self._freeze_backbone()

        # ------------------------------------------------------------------
        # Spatial attention
        # ------------------------------------------------------------------
        self.spatial_attention = SpatialAttention(
            in_channels=self._BACKBONE_OUT_CHANNELS,
            mid_channels=attention_mid_channels,
        )

        # ------------------------------------------------------------------
        # Projection head: adaptive pool -> flatten -> linear
        # ------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.projection = nn.Sequential(
            nn.Linear(self._BACKBONE_OUT_CHANNELS, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        logger.info(
            "SatelliteEncoder initialised: feature_dim=%d, pretrained=%s, "
            "freeze_backbone=%s",
            feature_dim,
            pretrained,
            freeze_backbone,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen.")

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        """Unfreeze backbone parameters starting from a given layer index.

        Layers are numbered 0–7, corresponding to ``[conv1, bn1, relu,
        maxpool, layer1, layer2, layer3, layer4]``.

        Args:
            from_layer: Index of the first layer to unfreeze (inclusive).
                Layers before this index stay frozen.
        """
        for idx, child in enumerate(self.backbone.children()):
            if idx >= from_layer:
                for param in child.parameters():
                    param.requires_grad = True
        logger.info("Backbone unfrozen from layer index %d onward.", from_layer)

    @property
    def num_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, satellite_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a satellite image into a feature vector with attention.

        Args:
            satellite_image: Input tensor of shape ``(B, 3, H, W)``.
                Expected to be normalised consistently with ResNet
                preprocessing (ImageNet mean/std).

        Returns:
            Tuple of:
                - features: Projected feature vector, shape ``(B, feature_dim)``.
                - attention_map: Spatial attention map, shape ``(B, 1, H', W')``
                  where ``H'`` and ``W'`` are the spatial dimensions of the
                  last convolutional feature map (e.g. 7×7 for 224×224 input).

        Raises:
            ValueError: If the input does not have 4 dimensions.
        """
        if satellite_image.ndim != 4:
            raise ValueError(
                f"Expected 4-D input (B, C, H, W), got {satellite_image.ndim}-D"
            )

        # Backbone feature extraction
        feature_map: torch.Tensor = self.backbone(satellite_image)  # (B, 2048, H', W')

        # Spatial attention
        attended_features, attention_map = self.spatial_attention(feature_map)

        # Global average pool + flatten
        pooled = self.global_pool(attended_features)  # (B, 2048, 1, 1)
        pooled = torch.flatten(pooled, start_dim=1)   # (B, 2048)

        # Project to feature_dim
        features = self.projection(pooled)  # (B, feature_dim)

        return features, attention_map
