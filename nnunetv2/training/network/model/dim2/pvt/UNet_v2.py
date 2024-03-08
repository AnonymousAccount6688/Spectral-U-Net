import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.network.model.dim2.utils import get_block, get_norm
from nnunetv2.training.network.model.dim2.unet_utils import inconv, down_block, up_block


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SDI(nn.Module):
    def __init__(self, channel, features):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3,
                       stride=1, padding=1)] * features)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class UNetV2(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=16,
                 block='FusedMBConv', pool=True,
                 deep_supervision=True,
                 channel=32):
        super().__init__()

        self.deep_supervision = deep_supervision
        block = get_block(block)
        nb = 2  # num_block

        self.inc = inconv(in_ch, base_ch, block=block)
        self.down1 = down_block(base_ch, 2 * base_ch, num_block=nb, block=block, pool=pool)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8 * base_ch, 16 * base_ch, num_block=nb, block=block, pool=pool)
        self.down5 = down_block(16 * base_ch, 32 * base_ch, num_block=nb, block=block, pool=pool)
        self.down6 = down_block(32 * base_ch, 32 * base_ch, num_block=nb, block=block, pool=pool)

        self.Translayer_1 = BasicConv2d(base_ch, channel, 1)
        self.Translayer_2 = BasicConv2d(2 * base_ch, channel, 1)
        self.Translayer_3 = BasicConv2d(4 * base_ch, channel, 1)
        self.Translayer_4 = BasicConv2d(8 * base_ch, channel, 1)
        self.Translayer_5 = BasicConv2d(16 * base_ch, channel, 1)
        self.Translayer_6 = BasicConv2d(32 * base_ch, channel, 1)
        self.Translayer_7 = BasicConv2d(32 * base_ch, channel, 1)

        self.sdi_1 = SDI(channel, 7)
        self.sdi_2 = SDI(channel, 7)
        self.sdi_3 = SDI(channel, 7)
        self.sdi_4 = SDI(channel, 7)
        self.sdi_5 = SDI(channel, 7)
        self.sdi_6 = SDI(channel, 7)
        self.sdi_7 = SDI(channel, 7)

        self.up0_0 = up_block(32, 32, num_block=nb, block=block)
        self.up0 = up_block(32, 32, num_block=nb, block=block)
        self.up1 = up_block(32, 32, num_block=nb, block=block)
        self.up2 = up_block(32, 32, num_block=nb, block=block)
        self.up3 = up_block(32, 32, num_block=nb, block=block)
        self.up4 = up_block(32, 32, num_block=nb, block=block)

        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

        seg_layers = []
        for i in range(6):
            seg_layers.append(nn.Conv2d(32, num_classes, 1, 1, 0,
                                        bias=True))

        self.seg_0 = nn.Conv2d(32, num_classes, 1, 1, 0, bias=True)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, x):
        x1 = self.inc(x)  # (H, W)
        x2 = self.down1(x1)  # (H/2, W/2)
        x3 = self.down2(x2)  # (H/4, W/4)
        x4 = self.down3(x3)  # (H/8, W/8)
        x5 = self.down4(x4)  # (H/16, W/16)
        x6 = self.down5(x5)  # (H/32, W/32)
        x7 = self.down6(x6)  # (H/64, W/64)

        x1 = self.Translayer_1(x1)
        x2 = self.Translayer_2(x2)
        x3 = self.Translayer_3(x3)
        x4 = self.Translayer_4(x4)
        x5 = self.Translayer_5(x5)
        x6 = self.Translayer_6(x6)
        x7 = self.Translayer_7(x7)

        x1 = self.sdi_1([x1, x2, x3, x4, x5, x6, x7], x1)
        x2 = self.sdi_2([x1, x2, x3, x4, x5, x6, x7], x2)
        x3 = self.sdi_3([x1, x2, x3, x4, x5, x6, x7], x3)
        x4 = self.sdi_4([x1, x2, x3, x4, x5, x6, x7], x4)
        x5 = self.sdi_5([x1, x2, x3, x4, x5, x6, x7], x5)
        x6 = self.sdi_6([x1, x2, x3, x4, x5, x6, x7], x6)
        x7 = self.sdi_7([x1, x2, x3, x4, x5, x6, x7], x7)

        seg_out = []

        out = self.up0_0(x7, x6)
        seg_out.append(self.seg_layers[0](out))

        out = self.up0(out, x5)  # 1st out, (H/16, W/16)
        seg_out.append(self.seg_layers[1](out))

        out = self.up1(out, x4)  # 2nd out, (H/8, W/8)
        seg_out.append(self.seg_layers[2](out))

        out = self.up2(out, x3)  # 3rd out,  (H/4, W/4)
        seg_out.append(self.seg_layers[3](out))

        out = self.up3(out, x2)  # 4th out, (H/2, W/ 2)
        seg_out.append(out)

        out = self.up4(out, x1)  # 5-th out  (C, H, W)
        out = self.outc(out)  # (cls, H, W)
        seg_out.append(out)

        if self.deep_supervision:
            return seg_out[::-1]
        else:
            return seg_out[-1]


if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))  # .cuda()
    # ConvNeXtBlock FusedMBConv
    net = UNetV2(in_ch=3, num_classes=1, block="FusedMBConv")  # .cuda()
    ys = net(x)
    print(len(ys))
    for y in ys:
        print(y.shape)
    # print(y[0].shape)
