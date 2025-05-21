# models/spectral_transform.py
import torch
import torch.nn as nn
import torch.fft

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # FFT 결과는 real + imag → 채널 2배로 취급
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
        """
        x: (B, C, H, W)
        Return: (B, out_channels, H, W)
        """

        # FFT → complex (B, C, H, W) → (B, C, H, W//2 + 1) 복소수
        fft_complex = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1), complex

        fft_real = fft_complex.real
        fft_imag = fft_complex.imag

        # Concatenate real & imag: (B, 2C, H, W//2+1)
        fft_feat = torch.cat([fft_real, fft_imag], dim=1)

        # conv → relu → conv → relu → conv
        out = self.conv1(fft_feat)
        out = self.conv2(out)
        out = self.final_conv(out)

        # Split real & imag
        out_c = out.shape[1] // 2
        real, imag = out[:, :out_c], out[:, out_c:]

        # 다시 complex tensor로
        fft_modified = torch.complex(real, imag)

        # iFFT2 → (B, C_out, H, W)
        out_spatial = torch.fft.irfft2(fft_modified, s=x.shape[2:], norm='ortho')

        return out_spatial
