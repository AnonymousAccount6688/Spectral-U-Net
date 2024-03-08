import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import get_block, get_norm
from layers import inconv, down_block, up_block


class UNet(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=32, block='SingleConv', pool=True):
        super().__init__()

        block = get_block(block)
        nb = 2  # num_block

        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2 * base_ch, num_block=nb, block=block, pool=pool)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8 * base_ch, 16 * base_ch, num_block=nb, block=block, pool=pool)
        # self.down5 = down_block(16 * base_ch, 32 * base_ch, num_block=nb, block=block, pool=pool)

        # self.up0 = up_block(32*base_ch, 16*base_ch, num_block=nb, block=block)
        self.up1 = up_block(16 * base_ch, 8 * base_ch, num_block=nb, block=block)
        self.up2 = up_block(8 * base_ch, 4 * base_ch, num_block=nb, block=block)
        self.up3 = up_block(4 * base_ch, 2 * base_ch, num_block=nb, block=block)
        self.up4 = up_block(2 * base_ch, base_ch, num_block=nb, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)  # no down-sample
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.down5(x5)

        # print(f"x6.shape: {x6.shape}")
        # out = self.up0(x6, x5)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)

        return out


if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256)).cuda()  # SingleConv
    net = UNet(in_ch=1, num_classes=2, block="BasicBlock").cuda()
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

    flops = FlopCountAnalysis(net, torch.randn(1, 1, 256, 256).cuda()).total()
    print(f"flops: {flops / 10 ** 9:.3f} G")

    params = parameter_count(net)['']
    print(f"params: {params / 10 ** 6:.3f} M")

    # summary(net, input_size=(1, 256, 256))
    # y = net(x)
    # print(y.shape)
