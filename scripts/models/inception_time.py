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
# 4. InceptionTime - State-of-the-art for time series classification
# =============================================================================


class InceptionModule1D(nn.Module):
    """Single Inception module for time series.

    Args:
        in_channels: Number of input channels
        n_filters: Number of filters per branch
        kernel_sizes: Kernel sizes for parallel convolutions
        bottleneck_channels: Bottleneck channels (0 = no bottleneck)
        use_residual: Whether to use residual connection
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels: int = 32,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual

        # Bottleneck
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            if bottleneck_channels > 0
            else None
        )

        # Parallel convolutions with different kernel sizes
        conv_in = bottleneck_channels if bottleneck_channels > 0 else in_channels
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in, n_filters, kernel_size, padding=kernel_size // 2, bias=False
                )
                for kernel_size in kernel_sizes
            ]
        )

        # MaxPool branch
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, n_filters, 1, bias=False)

        # Batch norm and activation
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

        # Residual connection
        if use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels, n_filters * (len(kernel_sizes) + 1), 1, bias=False
                ),
                nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1)),
            )

    def forward(self, x):
        # Bottleneck
        if self.bottleneck is not None:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        # Parallel convolutions
        conv_outputs = [conv(x_bottleneck) for conv in self.conv_list]

        # MaxPool branch
        pool_out = self.maxpool(x)
        pool_out = self.conv_pool(pool_out)

        # Concatenate all branches
        out = torch.cat(conv_outputs + [pool_out], dim=1)
        out = self.bn(out)

        # Residual connection
        if self.use_residual:
            out = out + self.residual(x)

        out = self.relu(out)
        return out


class InceptionTime(nn.Module):
    """InceptionTime: Ensemble of Inception-based networks for time series.

    State-of-the-art performance on UCR time series archive.

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        n_filters: Number of filters per inception branch
        depth: Number of inception modules
        kernel_sizes: Kernel sizes for parallel convolutions
        bottleneck_channels: Bottleneck channels
        dropout: Dropout probability

    Input:
        x: (B, C, T) where B=batch, C=channels, T=time

    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 2,
        n_filters: int = 32,
        depth: int = 6,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Stack of Inception modules
        self.inception_blocks = nn.ModuleList()

        in_channels = input_channels
        for i in range(depth):
            use_residual = True if i % 3 == 2 else False  # Residual every 3 blocks
            self.inception_blocks.append(
                InceptionModule1D(
                    in_channels,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual,
                )
            )
            in_channels = n_filters * (len(kernel_sizes) + 1)

        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, num_classes) logits
        """
        for block in self.inception_blocks:
            x = block(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
