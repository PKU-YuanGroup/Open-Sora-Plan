import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import shift_dim

class ChannelLayerNorm(nn.Module):
    # layer norm on channels
    def __init__(self, in_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.norm(x)
        x = shift_dim(x, -1, 1)
        return x


class NormReLU(nn.Module):

    def __init__(self, channels, relu=True, affine=True):
        super().__init__()

        self.relu = relu
        self.norm = ChannelLayerNorm(channels)

    def forward(self, x):
        x_float = x.float()
        x_float = self.norm(x_float)
        x = x_float.type_as(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, filters, stride, use_projection=False):
        super().__init__()

        if use_projection:
            self.proj_conv = nn.Conv3d(in_channels, filters, kernel_size=1,
                                       stride=stride, bias=False)
            self.proj_bnr = NormReLU(filters, relu=False)

        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bnr1 = NormReLU(filters)

        self.conv2 = nn.Conv3d(filters, filters, kernel_size=3,
                               stride=1, bias=False, padding=1)
        self.bnr2 = NormReLU(filters)

        self.use_projection = use_projection

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.proj_bnr(self.proj_conv(x))
        x = self.bnr1(self.conv1(x))
        x = self.bnr2(self.conv2(x))

        return F.relu(x + shortcut, inplace=True)

class BlockGroup(nn.Module):

    def __init__(self, in_channels, filters, blocks, stride):
        super().__init__()

        self.start_block = ResidualBlock(in_channels, filters, stride, use_projection=True)
        in_channels = filters

        self.blocks = []
        for _ in range(1, blocks):
            self.blocks.append(ResidualBlock(in_channels, filters, 1))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.start_block(x)
        x = self.blocks(x)
        return x


class ResNet(nn.Module):

    def __init__(self, in_channels, layers, width_multiplier,
                 stride, resnet_dim=240, cifar_stem=True):
        super().__init__()
        self.width_multiplier = width_multiplier
        self.resnet_dim = resnet_dim

        assert all([int(math.log2(d)) == math.log2(d) for d in stride]), stride
        n_times_downsample = np.array([int(math.log2(d)) for d in stride])

        if cifar_stem:
            self.stem = nn.Sequential(
                nn.Conv3d(in_channels, 64 * width_multiplier,
                          kernel_size=3, padding=1, bias=False),
                NormReLU(64 * width_multiplier)
            )
        else:
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            n_times_downsample -= 1  # conv
            n_times_downsample[-2:] = n_times_downsample[-2:] - 1  # pooling
            self.stem = nn.Sequential(
                nn.Conv3d(in_channels, 64 * width_multiplier,
                          kernel_size=7, stride=stride, bias=False,
                          padding=3),
                NormReLU(64 * width_multiplier),
                nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
            )

        self.group1 = BlockGroup(64 * width_multiplier, 64 * width_multiplier,
                                 blocks=layers[0], stride=1)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group2 = BlockGroup(64 * width_multiplier, 128 * width_multiplier,
                                 blocks=layers[1], stride=stride)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group3 = BlockGroup(128 * width_multiplier, 256 * width_multiplier,
                                 blocks=layers[2], stride=stride)

        stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
        n_times_downsample -= 1
        self.group4 = BlockGroup(256 * width_multiplier, resnet_dim,
                                 blocks=layers[3], stride=stride)
        assert all([d <= 0 for d in n_times_downsample]), f'final downsample {n_times_downsample}'

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = shift_dim(x, 1, -1)

        return x


def resnet34(width_multiplier, stride, cifar_stem=True, resnet_dim=240):
    return ResNet(3, [3, 4, 6, 3], width_multiplier,
                  stride, cifar_stem=cifar_stem, resnet_dim=resnet_dim)
