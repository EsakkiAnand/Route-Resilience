"""
Baseline U-Net Architecture Implementation in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Baseline U-Net model with 4 encoder-decoder depth levels."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, init_features: int = 32):
        super().__init__()
        features = init_features

        # Encoder blocks
        self.inc = DoubleConv(in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features, features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 2, features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 4, features * 8))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 8, features * 16))

        # Decoder blocks
        self.up1 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features * 16, features * 8)

        self.up2 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features * 8, features * 4)

        self.up3 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features * 4, features * 2)

        self.up4 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(features * 2, features)

        # Output head
        self.outc = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        bn = self.bottleneck(x4)

        # Decoder pass with skip connections
        d1 = self.up1(bn)
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits
