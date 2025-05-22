# models/spectral_transform.py
import torch
import torch.nn as nn

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1)

    def forward(self, x):
        fft_complex = torch.fft.rfft2(x, norm='ortho')
        fft_feat = torch.cat([fft_complex.real, fft_complex.imag], dim=1)
        out = self.conv1(fft_feat)
        out = self.conv2(out)
        out = self.final_conv(out)
        real, imag = torch.chunk(out, 2, dim=1)
        out_complex = torch.complex(real, imag)
        return torch.fft.irfft2(out_complex, s=x.shape[2:], norm='ortho')
