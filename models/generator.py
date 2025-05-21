# models/generator.py
import torch.nn as nn
from .residual_block import FFCResBlock

class LaMaGenerator(nn.Module):
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        )
        self.body = nn.Sequential(*[FFCResBlock(base_channels, base_channels) for _ in range(9)])
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.body(x)
        x = self.up(x)
        return x
