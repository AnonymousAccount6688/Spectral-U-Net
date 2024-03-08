from torch import nn
import torch
import torch.nn.functional as F
from nnunetv2.training.network.model.waveunet.torch_wavelets import DWT_2D, IDWT_2D


class Downsample(nn.Module):
    def __init__(self, in_channels, channel_expansion=2):
        super().__init__()
        if channel_expansion == 2:
            self.linear = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1)
        elif channel_expansion == 1:
            self.linear = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.dwt = DWT_2D(wave='haar')

    def forward(self, x):
        x = self.linear(x)
        x = self.dwt(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=4, act=nn.ReLU):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, kernel_size=1),
            act(),
            nn.Conv2d(in_ch // ratio, in_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = self.excitation(out)

        return x * out


class DropPath(nn.Module):
    """
    Drop connection with pobability p
    """

    def __init__(self, p=0):
        super().__init__()

        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1).to(x.device)
        binary_mask = self.p < random_tensor

        x = x.div(1 - self.p)
        x = x * binary_mask

        return x


class MBConv(nn.Module):
    """
    MBConv with an expansion factor of N, and squeeze-and-excitation module
    """

    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch
        self.se = se

        self.expand_proj = nn.Identity() if (expansion == 1) else ConvNormAct(in_ch, expanded, kernel_size=1, norm=norm,
                                                                              act=act, preact=True)

        self.depthwise = ConvNormAct(expanded, expanded, kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=expanded, act=act, norm=norm, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                ConvNormAct(in_ch, out_ch, kernel_size, stride=stride, padding=padding, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.expand_proj(x)
        x = self.depthwise(x)
        if self.se:
            x = self.se_block(x)
        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x


class FusedMBConv(nn.Module):
    """
    MBConv with an expansion factor of N, and squeeze-and-excitation module
    """

    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1, ratio=4, p=0, se=True, norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super().__init__()

        padding = (kernel_size - 1) // 2
        expanded = expansion * in_ch

        self.stride = stride
        self.se = se

        self.conv3x3 = ConvNormAct(in_ch, expanded, kernel_size=kernel_size, stride=stride, padding=padding, groups=1,
                                   norm=norm, act=act, preact=True)

        if self.se:
            self.se_block = SEBlock(expanded, ratio=ratio)

        self.pointwise = ConvNormAct(expanded, out_ch, kernel_size=1, padding=0, norm=norm, act=False, preact=True)

        self.drop_path = DropPath(p)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=False, act=False))

    def forward(self, x):
        residual = x

        x = self.conv3x3(x)
        if self.se:
            x = self.se_block(x)
        x = self.pointwise(x)

        x = self.drop_path(x)

        x = x + self.shortcut(residual)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=1, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        self.expansion = 4
        self.conv1 = ConvNormAct(in_ch, out_ch // self.expansion, 1, stride=1, padding=0, norm=norm, act=act,
                                 preact=preact)
        self.conv2 = ConvNormAct(out_ch // self.expansion, out_ch // self.expansion, 3, stride=stride, padding=1,
                                 norm=norm, act=act, groups=groups, dilation=dilation, preact=preact)

        self.conv3 = ConvNormAct(out_ch // self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act,
                                 preact=preact)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shortcut(residual)

        return out


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False, norm=nn.BatchNorm2d,
                 act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        if stride == 2:
            self.conv = Downsample(in_channels=in_ch)
        else:
            self.conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=bias
            )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):

        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.

    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with

    shape (batch_size, height, width, channels) while channels_first corresponds to inputs

    with shape (batch_size, channels, height, width).

    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        return self.conv(x)


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, out_ch, stride=1, kernel_size=7, norm=None, act=None, preact=None, drop_path=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


def get_block(name):
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'MBConv': MBConv,
        'FusedMBConv': FusedMBConv,
        'ConvNeXtBlock': ConvNeXtBlock
    }
    return block_map[name]


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, block=BasicBlock):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = block(out_ch, out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, pool=True):
        super().__init__()

        block_list = []

        if pool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x):
        return self.conv(x)


class WaveBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, pool=True, channel_expansion=2):
        super().__init__()
        block_list = []
        if pool:
            block_list.append(Downsample(in_ch, channel_expansion=channel_expansion))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        # self.conv = nn.Sequential(*block_list)
        self.conv = nn.ModuleList(block_list)

    def forward(self, x):
        ans = []
        for mou in self.conv:
            x = mou(x)
            ans.append(x)
        # return self.conv(x)

        return ans


class iwave_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock):
        super().__init__()

        # self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self.idwt1 = IDWT_2D(wave='haar')
        self.idwt2 = IDWT_2D(wave='haar')

        block_list = []
        block_list.append(block(in_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        if isinstance(x2, (list, tuple)):
            x2 = x2[1]
        """
            x2 is from skip connection. x2 is a tuple of tensor

        :param x1:
        :param x2:
        :return:
        """
        # x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        # x1 = self.conv_ch(x1)  # reduce the channel

        x10 = self.idwt1(x1[0])
        x11 = self.idwt2(x1[1])

        out = torch.cat([x2, x10, x11], dim=1)
        # out = torch.cat([x2, x1], dim=1)

        ans = []
        for mou in self.conv:
            out = mou(out)
            ans.append(out)

        return ans
        # out = self.conv(out)  # reduce the  channel
        #
        # return out


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock):
        super().__init__()

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        block_list = []
        block_list.append(block(2*out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        """
            x2 is from skip connection

        :param x1:
        :param x2:
        :return:
        """
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)  # reduce the channel

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)  # reduce the  channel

        return out


def get_norm(name):
    norm_map = {'bn': nn.BatchNorm3d,
                'in': nn.InstanceNorm3d
                }

    return norm_map[name]