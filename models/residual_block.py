# models/residual_block.py
import torch.nn as nn
from .ffc import FFC

class FFCResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            FFC(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            FFC(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.block(x) + x)
