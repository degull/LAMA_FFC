import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 2)
        )

    def forward(self, x):
        return self.net(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([PatchDiscriminator() for _ in range(num_scales)])

    def forward(self, x):
        outputs = []
        for i, D in enumerate(self.discriminators):
            out = D(x)
            outputs.append(out)
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)  # Downsample
        return outputs
