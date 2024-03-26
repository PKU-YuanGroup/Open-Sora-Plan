from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import cast_tuple
from .conv import CausalConv3d


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1, ) + stride,
            padding=0
        )

    def forward(self, x):
        pad = (0,1,0,1,0,0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class SpatialUpsample2x(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.chan_in,
            self.chan_out,
            (1,) + self.kernel_size,
            stride=(1, ) + stride,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1,2,2), mode="nearest")
        x = self.conv(x)
        return x
    
class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.AvgPool3d((kernel_size,1,1), stride=(2,1,1))
        
    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.kernel_size - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return self.conv(x)

class TimeUpsample2x(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    def forward(self, x):
        if x.size(2) > 1:
            x,x_= x[:,:,:1],x[:,:,1:]
            x_= F.interpolate(x_, scale_factor=(2,1,1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        return x

# class TimeDownsample2x(nn.Module):
#     def __init__(
#         self,
#         chan_in,
#         chan_out,
#         kernel_size: int = 3,
#         padding: int = 1
#     ):
#         super().__init__()
#         self.chan_in = chan_in
#         self.chan_out = chan_out
#         self.kernel_size = kernel_size
#         self.conv = CausalConv3d(chan_in, chan_out, kernel_size, stride=(2, 1, 1), padding=padding)
        
#     def forward(self, x):
#         return self.conv(x)

# class TimeUpsample2x(nn.Module):
#     def __init__(
#         self,
#         chan_in,
#         chan_out,
#         kernel_size: int = 3,
#     ):
#         super().__init__()
#         self.chan_in = chan_in
#         self.chan_out = chan_out
#         self.kernel_size = kernel_size
#         self.conv = CausalConv3d(chan_in, chan_out, kernel_size, stride=1, padding=(0,1,1))
        
#     def forward(self, x):
#         if x.size(2) > 1:
#             x,x_= x[:,:,:1],x[:,:,1:]
#             x = F.interpolate(x, scale_factor=(1,1,1), mode='nearest')
#             x_= F.interpolate(x_, scale_factor=(2,1,1), mode='nearest')
#             x = torch.concat([x, x_], dim=2)
#         x = self.conv(x)
#         return x
    