"""
Neural network model architectures for time series classification.

This module provides several state-of-the-art models:
- ImprovedCNN1D: Deep CNN with residual-like connections
- TimeSeriesTransformer: Transformer encoder for time series
- ResNet1D: 1D ResNet adapted from image classification
- InceptionTime: Multi-scale inception modules
- RocketClassifier: Random convolutional kernel features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 2. TimeSeriesTransformer - Transformer encoder for time series
# =============================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x: (B, T, d_model) input tensor

        Returns:
            (B, T, d_model) tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for time series classification.

    Args:
        input_dim: Number of input features (channels)
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_ff: Feedforward dimension
        dropout: Dropout probability

    Input:
        x: (B, T, C) where B=batch, T=time, C=channels

    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, C) input tensor

        Returns:
            (B, num_classes) logits
        """
        h = self.proj(x)  # (B, T, d_model)
        h = self.pos(h)
        h = self.encoder(h)  # (B, T, d_model)
        h = h.mean(dim=1)  # simple pooling over time (B, d_model)
        return self.cls_head(h)  # (B, 2)
