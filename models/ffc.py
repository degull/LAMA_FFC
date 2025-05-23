# models/ffc.py (Fast Fourier Convolution)
import torch
import torch.nn as nn
from .spectral_transform import SpectralTransform

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, ratio_gin=0.5, ratio_gout=0.5):
        super().__init__()

        # 채널 분할 비율에 따른 local/global 채널 수 계산
        self.in_cg = int(in_channels * ratio_gin)
        self.in_cl = in_channels - self.in_cg
        self.out_cg = int(out_channels * ratio_gout)
        self.out_cl = out_channels - self.out_cg

        # Local path: local → local
        self.conv_l2l = nn.Conv2d(self.in_cl, self.out_cl, kernel_size=3, padding=1)
        self.bn_l = nn.BatchNorm2d(self.out_cl)
        self.relu_l = nn.ReLU(inplace=True)

        # Local → global
        self.conv_l2g = nn.Conv2d(self.in_cl, self.out_cg, kernel_size=3, padding=1)

        # Global → local
        self.conv_g2l = nn.Conv2d(self.in_cg, self.out_cl, kernel_size=3, padding=1)

        # Global → global: spectral transform
        self.spectral = SpectralTransform(self.in_cg, self.out_cg)

        # Global branch BN + ReLU
        self.bn_g = nn.BatchNorm2d(self.out_cg)
        self.relu_g = nn.ReLU(inplace=True)

    def forward(self, x):
        # 정확한 채널 분할 (ratio 기반)
        x_l, x_g = torch.split(x, [self.in_cl, self.in_cg], dim=1)

        # Local output: local→local + global→local
        out_l = self.conv_l2l(x_l) + self.conv_g2l(x_g)
        out_l = self.relu_l(self.bn_l(out_l))

        # Global output: local→global + spectral(global→global)
        out_g = self.conv_l2g(x_l) + self.spectral(x_g)
        out_g = self.relu_g(self.bn_g(out_g))

        # 최종 출력 결합 (채널 기준 concat)
        return torch.cat([out_l, out_g], dim=1)
