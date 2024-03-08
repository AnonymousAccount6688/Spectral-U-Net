from torch_wavelets import *
from torch import nn
import torch


class WaveBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        self.idwt = IDWT_2D(wave='haar')

    def forward(self, x):
        # x = self.linear(x)
        x = self.idwt(x)  # (H, W, C) -> (2H, 2W, C/2)

        return x


if __name__ == "__main__":
    net = WaveBlock(16)
    x = torch.rand((2, 16, 32, 32))
    y = net(x)
    print(y.shape)