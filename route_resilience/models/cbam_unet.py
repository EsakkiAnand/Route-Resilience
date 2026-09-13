"""
CBAM Attention-Enhanced U-Net Architecture Implementation in PyTorch.
Embeds Channel & Spatial Attention Modules on skip connection features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from route_resilience.models.unet import DoubleConv, UNet


class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(in_channels // reduction_ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class CBAMBlock(nn.Module):
    """Combines Channel & Spatial Attention sequentially."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CBAMUNet(nn.Module):
    """U-Net with CBAM Attention Modules applied on skip connection paths."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, init_features: int = 32):
        super().__init__()
        features = init_features

        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features, features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 2, features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 4, features * 8))

        # CBAM Attention Gates on Skip Connections
        self.cbam1 = CBAMBlock(features)
        self.cbam2 = CBAMBlock(features * 2)
        self.cbam3 = CBAMBlock(features * 4)
        self.cbam4 = CBAMBlock(features * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(features * 8, features * 16))

        # Decoder
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        bn = self.bottleneck(x4)

        # Apply CBAM attention to skip features
        x4_att = self.cbam4(x4)
        x3_att = self.cbam3(x3)
        x2_att = self.cbam2(x2)
        x1_att = self.cbam1(x1)

        d1 = self.up1(bn)
        d1 = torch.cat([x4_att, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x3_att, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([x1_att, d4], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits


def get_model(architecture: str = "unet_resnet34", in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    """Factory function instantiating model architecture by config choice string.

    Args:
        architecture: 'unet_resnet34' (Baseline UNet) or 'cbam_unet' (CBAM Attention UNet).
        in_channels: Input image channels (3).
        num_classes: Output target classes (1).

    Returns:
        Instantiated PyTorch nn.Module.
    """
    arch_lower = architecture.lower()
    if "cbam" in arch_lower:
        return CBAMUNet(in_channels=in_channels, num_classes=num_classes)
    else:
        return UNet(in_channels=in_channels, num_classes=num_classes)
