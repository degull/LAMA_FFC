# models/spectral_transform.py
import torch
import torch.nn as nn

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 주파수 입력은 복소수 → 2배 채널 (Re + Im)
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
        self.final_fft_conv = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1)
        
        # 공간 도메인 후처리 Conv 1x1
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. FFT: 공간 → 주파수
        fft_complex = torch.fft.rfft2(x, norm='ortho')  # shape: (B, C, H, W//2+1), dtype=complex

        # 2. Re, Im 나눠서 concat
        fft_feat = torch.cat([fft_complex.real, fft_complex.imag], dim=1)  # shape: (B, 2C, H, W//2+1)

        # 3. 주파수 도메인에서 conv
        out = self.conv1(fft_feat)
        out = self.conv2(out)
        out = self.final_fft_conv(out)

        # 4. 다시 복소수로 합치기
        real, imag = torch.chunk(out, 2, dim=1)
        out_complex = torch.complex(real, imag)  # shape: (B, C, H, W//2+1)

        # 5. 역 FFT (주파수 → 공간)
        out_spatial = torch.fft.irfft2(out_complex, s=(H, W), norm='ortho')  # shape: (B, C, H, W)

        # 6. 마지막 Conv 1x1 (공간 도메인에서 채널 정제)
        return self.output_conv(out_spatial)
