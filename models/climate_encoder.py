"""Climate Time-Series Encoder with Temporal Attention.

Bidirectional LSTM encoder for multi-decadal climate data. Learns which
historical years contribute most to extinction risk via a temporal
attention mechanism, producing a fixed-length feature vector suitable
for downstream fusion.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) temporal attention over LSTM hidden states.

    Learns a scalar importance weight for each time step, allowing the
    model to focus on the years most predictive of extinction risk
    (e.g., recent deforestation events or climate anomalies).

    Args:
        hidden_dim: Dimensionality of each LSTM hidden state that will be
            attended over.  For a bidirectional LSTM this should be
            ``2 * lstm_hidden_dim``.
        attention_dim: Dimensionality of the internal attention projection.
            Defaults to ``hidden_dim // 2``.

    Shape:
        - Input: ``(B, T, hidden_dim)``
        - Output:
            - context: ``(B, hidden_dim)``
            - weights: ``(B, T)``  (sums to 1 over T)
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if attention_dim is None:
            attention_dim = max(hidden_dim // 2, 1)

        self.query = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

        # Learnable context query vector (global query)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))
        nn.init.normal_(self.context_vector, mean=0.0, std=1.0 / math.sqrt(attention_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            hidden_states: LSTM outputs of shape ``(B, T, hidden_dim)``.
            mask: Optional boolean mask of shape ``(B, T)`` where ``True``
                indicates a valid (non-padded) time step.  Padded positions
                receive ``-inf`` attention energy before softmax.

        Returns:
            Tuple of:
                - context: Weighted sum of hidden states, ``(B, hidden_dim)``.
                - attention_weights: Normalised attention scores, ``(B, T)``.
        """
        # Project hidden states
        projected = torch.tanh(self.query(hidden_states))  # (B, T, attn_dim)

        # Score each time step against the learnable context vector
        scores = torch.sum(projected * self.context_vector, dim=-1)  # (B, T)

        # Mask padded positions
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)  # (B, T)

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, T)
            hidden_states,                   # (B, T, H)
        ).squeeze(1)                         # (B, H)

        return context, attention_weights


class ClimateEncoder(nn.Module):
    """Bidirectional LSTM encoder for climate time-series data.

    Processes 30 years of monthly climate variables through a 2-layer
    bidirectional LSTM, then applies temporal attention to produce a
    fixed-length feature vector.

    Args:
        input_dim: Number of climate variables per time step (e.g. 12
            monthly measurements: temperature, precipitation, humidity, …).
        hidden_dim: Hidden size of each LSTM direction.  The effective
            hidden size is ``hidden_dim * 2`` due to bidirectionality.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers and on
            the final feature vector.
        seq_len: Expected sequence length (number of years).  Used only
            for documentation / assertions; not hard-coded into the
            architecture.
        attention_dim: Internal dimensionality of the temporal attention
            module.

    Example::

        encoder = ClimateEncoder()
        x = torch.randn(8, 30, 12)  # 8 samples, 30 years, 12 variables
        features, attn_weights = encoder(x)
        # features.shape     -> (8, 512)  (hidden_dim * 2)
        # attn_weights.shape  -> (8, 30)
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 30,
        attention_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_dim = hidden_dim * 2  # bidirectional

        # ------------------------------------------------------------------
        # Input projection (optional layer norm for stable training)
        # ------------------------------------------------------------------
        self.input_norm = nn.LayerNorm(input_dim)

        # ------------------------------------------------------------------
        # Bidirectional LSTM
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------
        # Temporal attention
        # ------------------------------------------------------------------
        self.temporal_attention = TemporalAttention(
            hidden_dim=self.output_dim,
            attention_dim=attention_dim,
        )

        # ------------------------------------------------------------------
        # Output dropout
        # ------------------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout)

        # Initialise LSTM weights
        self._init_weights()

        logger.info(
            "ClimateEncoder initialised: input_dim=%d, hidden_dim=%d, "
            "num_layers=%d, output_dim=%d",
            input_dim,
            hidden_dim,
            num_layers,
            self.output_dim,
        )

    def _init_weights(self) -> None:
        """Apply orthogonal initialisation to LSTM weight matrices."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 for stable long-range gradients
                hidden = self.hidden_dim
                param.data[hidden : 2 * hidden].fill_(1.0)

    @property
    def num_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        climate_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a climate time-series into a feature vector.

        Args:
            climate_seq: Climate data of shape ``(B, T, input_dim)``.
                Typically ``T=30`` (years) and ``input_dim=12`` (monthly
                climate variables).
            mask: Optional boolean mask of shape ``(B, T)`` indicating
                valid time steps (``True`` = valid).  Useful when
                sequences are padded to uniform length.

        Returns:
            Tuple of:
                - features: Context vector from temporal attention,
                  shape ``(B, output_dim)`` where ``output_dim = 2 * hidden_dim``.
                - attention_weights: Per-time-step attention scores,
                  shape ``(B, T)``.

        Raises:
            ValueError: If input is not 3-D.
        """
        if climate_seq.ndim != 3:
            raise ValueError(
                f"Expected 3-D input (B, T, input_dim), got {climate_seq.ndim}-D"
            )

        # Normalise input features
        x = self.input_norm(climate_seq)  # (B, T, input_dim)

        # BiLSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim * 2)

        # Temporal attention
        context, attention_weights = self.temporal_attention(
            lstm_out, mask=mask
        )  # context: (B, output_dim), weights: (B, T)

        # Apply dropout
        features = self.dropout(context)

        return features, attention_weights
