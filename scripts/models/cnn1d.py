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
# 1. ImprovedCNN1D - Deep CNN with improved regularization
# =============================================================================


class ImprovedCNN1D(nn.Module):
    """Improved CNN1D with deeper architecture and better regularization.

    Features:
    - 6 convolutional blocks with learned downsampling (stride-2 conv)
    - Less aggressive pooling (keeps 64 timesteps before final pool)
    - Spatial dropout for regularization
    - Gradual channel expansion

    Args:
        C: Number of input channels
        dropout: Dropout probability for classification head

    Input:
        x: (B, C, T) where B=batch, C=channels, T=time steps

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, C: int, dropout: float = 0.3):
        super().__init__()

        # Block 1: 512 -> 512 (no pooling)
        self.conv1 = nn.Sequential(
            nn.Conv1d(C, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Block 2: 512 -> 256 (stride-2 conv for learned downsampling)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )

        # Block 3: 256 -> 256 (no pooling, feature refinement)
        self.conv3 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )

        # Block 4: 256 -> 128 (stride-2 conv)
        self.conv4 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Block 5: 128 -> 128 (no pooling, feature refinement)
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Block 6: 128 -> 64 (stride-2 conv)
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
        )

        # Global pooling: 64 timesteps -> 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head with additional regularization
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, num_classes) logits
        """
        x = self.conv1(x)  # (B, 64, 512)
        x = self.conv2(x)  # (B, 96, 256)
        x = self.conv3(x)  # (B, 96, 256) - refinement
        x = self.conv4(x)  # (B, 128, 128)
        x = self.conv5(x)  # (B, 128, 128) - refinement
        x = self.conv6(x)  # (B, 192, 64)

        x = self.global_pool(x)  # (B, 192, 1)
        x = x.squeeze(-1)  # (B, 192)

        return self.head(x)  # (B, 2)
