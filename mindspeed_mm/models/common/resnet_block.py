import torch
from torch import nn

from mindspeed_mm.utils.utils import video_to_image
from .conv import CausalConv3d
from ..common.normalize import Normalize

def nonlinearity(x):
    return x * torch.sigmoid(x)

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        dropout,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        conv_shortcut=False,
        num_groups=32,
        eps=1e-6,
        affine=True,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.norm2 = Normalize(out_channels, num_groups, eps, affine, norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    @video_to_image
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        x = x + h
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
        conv_shortcut=False,
        dropout=0,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        self.norm2 = Normalize(out_channels, num_groups, eps, affine, norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size, padding=padding)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size, padding=padding)
            else:
                self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h