import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.network.model.unet.layers import get_block, get_norm
from nnunetv2.training.network.model.unet.layers import inconv, down_block, up_block


class UNet(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=32, block='SingleConv', pool=True,
                 deep_supervision=True):
        super().__init__()

        self.deep_supervision = deep_supervision

        block = get_block(block)
        nb = 2  # num_block

        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2 * base_ch, num_block=nb, block=block, pool=pool)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8 * base_ch, 16 * base_ch, num_block=nb, block=block, pool=pool)
        self.down5 = down_block(16 * base_ch, 16 * base_ch, num_block=nb, block=block, pool=pool)
        self.down6 = down_block(16 * base_ch, 16 * base_ch, num_block=nb, block=block, pool=pool)

        self.up0_0 = up_block(16*base_ch, 16*base_ch, num_block=nb, block=block)
        self.up0 = up_block(16*base_ch, 16*base_ch, num_block=nb, block=block)
        self.up1 = up_block(16 * base_ch, 8 * base_ch, num_block=nb, block=block)
        self.up2 = up_block(8 * base_ch, 4 * base_ch, num_block=nb, block=block)
        self.up3 = up_block(4 * base_ch, 2 * base_ch, num_block=nb, block=block)
        self.up4 = up_block(2 * base_ch, base_ch, num_block=nb, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

        self.segouts = nn.ModuleList([
            nn.Conv2d(16 * base_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(16 * base_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(8 * base_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(4 * base_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(2 * base_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True),

        ])


    def forward(self, x):
        x1 = self.inc(x)  # no down-sample
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        # print(f"x6.shape: {x6.shape}")
        segout = []
        out = self.up0_0(x7, x6)
        segout.append(self.segouts[0](out))

        out = self.up0(out, x5)
        segout.append(self.segouts[1](out))
        # out = self.up0(x6, x5)
        out = self.up1(out, x4)
        segout.append(self.segouts[2](out))

        out = self.up2(out, x3)
        segout.append(self.segouts[3](out))

        out = self.up3(out, x2)
        segout.append(self.segouts[4](out))

        out = self.up4(out, x1)
        # segout.append(self.segouts[5](out))

        out = self.outc(out)
        segout.append(out)

        if self.deep_supervision:
            return segout[::-1]
        else:
            return out


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))  # .cuda().float()  # SingleConv
    net = UNet(in_ch=3, num_classes=2, block="SingleConv")  # .cuda()
    ys = net(x)
    for y in ys:
        print(y.shape)
