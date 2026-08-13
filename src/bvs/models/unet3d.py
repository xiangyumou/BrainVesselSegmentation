from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class StandardUNet3D(nn.Module):
    """Four-level 3D U-Net comparison baseline."""

    def __init__(
        self, in_channels: int = 1, out_channels: int = 2, base_channels: int = 32
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.pool = nn.MaxPool3d(2)
        self.dec4 = UpBlock(c * 16, c * 8, c * 8)
        self.dec3 = UpBlock(c * 8, c * 4, c * 4)
        self.dec2 = UpBlock(c * 4, c * 2, c * 2)
        self.dec1 = UpBlock(c * 2, c, c)
        self.head = nn.Conv3d(c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        features = self.bottleneck(self.pool(e4))
        x = self.dec4(features, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        logits = self.head(x)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
            "features": features,
        }
