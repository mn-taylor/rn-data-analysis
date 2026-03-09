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
# 3. ResNet1D - Adapted from pretrained ResNet for 1D time series
# =============================================================================


class ResNetBlock1D(nn.Module):
    """1D ResNet block with skip connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        stride: Stride
        downsample: Optional downsample layer for skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """ResNet adapted for 1D time series.

    Can use transfer learning from image-based ResNet.

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        dropout: Dropout probability

    Input:
        x: (B, C, T) where B=batch, C=channels, T=time

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, input_channels: int, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        # Initial conv layer
        self.conv1 = nn.Conv1d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(
            ResNetBlock1D(in_channels, out_channels, stride=stride, downsample=downsample)
        )
        for _ in range(1, blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, num_classes) logits
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
