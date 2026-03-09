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
# 5. ROCKET - Random Convolutional Kernel Transform
# =============================================================================


class RocketFeatures(nn.Module):
    """ROCKET: Random convolutional kernels for fast time series classification.

    Generates random kernels and extracts max/PPV features.

    Args:
        input_channels: Number of input channels
        num_kernels: Number of random kernels
        kernel_sizes: Possible kernel sizes
    """

    def __init__(
        self, input_channels: int, num_kernels: int = 10000, kernel_sizes=[7, 9, 11]
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_sizes = kernel_sizes

        # Generate random kernels
        self.kernels = nn.ModuleList()
        self.biases = nn.ParameterList()
        self.dilations = []
        self.paddings = []

        for _ in range(num_kernels):
            kernel_size = kernel_sizes[torch.randint(0, len(kernel_sizes), (1,)).item()]
            dilation = 2 ** torch.randint(0, 5, (1,)).item()  # Random dilation
            padding = (kernel_size - 1) * dilation // 2

            # Random conv kernel (frozen - not trained)
            conv = nn.Conv1d(
                input_channels, 1, kernel_size, dilation=dilation, padding=padding, bias=False
            )
            nn.init.normal_(conv.weight)

            # Freeze kernel weights
            for param in conv.parameters():
                param.requires_grad = False

            self.kernels.append(conv)
            self.biases.append(nn.Parameter(torch.randn(1) * 0.1, requires_grad=False))
            self.dilations.append(dilation)
            self.paddings.append(padding)

    def forward(self, x):
        """Extract ROCKET features.

        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, num_kernels * 2) features
        """
        features = []

        for conv, bias in zip(self.kernels, self.biases):
            # Apply convolution
            out = conv(x)  # (B, 1, T)
            out = out.squeeze(1)  # (B, T)
            out = out + bias

            # Extract features: max and proportion of positive values (PPV)
            max_val = torch.max(out, dim=1)[0]  # (B,)
            ppv = torch.mean((out > 0).float(), dim=1)  # (B,)

            features.append(max_val)
            features.append(ppv)

        return torch.stack(features, dim=1)  # (B, num_kernels * 2)


class RocketClassifier(nn.Module):
    """ROCKET-based classifier: Random kernels + simple linear classifier.

    Very fast and effective for small datasets.

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        num_kernels: Number of random kernels
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
        num_kernels: int = 5000,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.rocket = RocketFeatures(input_channels, num_kernels=num_kernels)

        # Simple linear classifier on top of ROCKET features
        feature_dim = num_kernels * 2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, num_classes) logits
        """
        features = self.rocket(x)
        return self.classifier(features)
