# models/ffc.py (Fast Fourier Convolution)
import torch
import torch.nn as nn
from .spectral_transform import SpectralTransform

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, ratio_gin=0.5, ratio_gout=0.5):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.conv_l2l = nn.Conv2d(in_cl, out_cl, kernel_size=3, padding=1)
        self.conv_l2g = nn.Conv2d(in_cl, out_cg, kernel_size=3, padding=1)
        self.conv_g2l = nn.Conv2d(in_cg, out_cl, kernel_size=3, padding=1)
        self.spectral = SpectralTransform(in_cg, out_cg)

    def forward(self, x):
        x_l, x_g = torch.split(x, [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2], dim=1)
        out_l = self.conv_l2l(x_l) + self.conv_g2l(x_g)
        out_g = self.conv_l2g(x_l) + self.spectral(x_g)
        return torch.cat([out_l, out_g], dim=1)
