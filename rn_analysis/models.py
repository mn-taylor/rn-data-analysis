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
