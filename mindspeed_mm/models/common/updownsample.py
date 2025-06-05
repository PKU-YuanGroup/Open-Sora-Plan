from typing import Union, Tuple
from collections import deque

import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from mindspeed_mm.utils.utils import cast_tuple, video_to_image
from .conv import CausalConv3d, WfCausalConv3d


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    @video_to_image
    def forward(self, x, scale_factor=2.0, mode="nearest"):
        x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        undown=False
    ):
        super().__init__()
        self.undown = undown
        # no asymmetric padding in torch conv, must do it ourselves
        if self.undown:
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=1
            )
        else:
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=0
            )

    @video_to_image
    def forward(self, x):
        if self.undown:
            x = self.conv(x)
        else:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        return x


class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.in_channels,
            self.out_channels,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class SpatialUpsample2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
        unup=False,
    ):
        super().__init__()
        self.unup = unup
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            (1,) + kernel_size,
            stride=(1,) + stride,
            padding=1
        )

    def forward(self, x):
        if not self.unup:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> b (c t) h w")
            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
            x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 2
    ):
        super().__init__()
        # ori: self.conv = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))
        # note: when kernel_size=(kernel_size, 1, 1), and stride=(stride, 1, 1), can be replaced by pool1d
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.kernel_size - 1, 1, 1))
        x = torch.concatenate((first_frame_pad, x), dim=2)
        n, c, d, h, w = x.shape
        x = torch_npu.npu_confusion_transpose(x, (0, 1, 3, 4, 2), (n, c * h * w, d), True)
        conv_res = self.conv(x)
        b, s, m = conv_res.shape
        conv_res = torch_npu.npu_confusion_transpose(conv_res, (0, 1, 4, 2, 3),
                                                     (n, c, h, w, (b * s * m) // (n * c * h * w)), False)
        return conv_res


class TimeUpsample2x(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.size(2) > 1:
            x, y = x[:, :, :1], x[:, :, 1:]
            y = F.interpolate(y, scale_factor=(2, 1, 1), mode="trilinear")
            x = torch.concat([x, y], dim=2)
        return x


class TimeDownsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: Tuple[int] = (2, 1, 1),
        padding: Tuple[int] = (0, 1, 1),
        mix_factor: float = 2.0
    ):
        super().__init__()
        # ori: self.conv = nn.AvgPool3d((kernel_size, 1, 1), stride=(2, 1, 1))
        # note: when kernel_size=(kernel_size, 1, 1), and stride=(stride, 1, 1), can be replaced by pool1d
        self.avg_pool = nn.AvgPool1d(kernel_size, stride[0])
        kernel_size = cast_tuple(kernel_size, 3)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.kernel_size[0] - 1, 1, 1))
        x = torch.concatenate((first_frame_pad, x), dim=2)
        n, c, d, h, w = x.shape
        x = torch_npu.npu_confusion_transpose(x, (0, 1, 3, 4, 2), (n, c * h * w, d), True)
        pool_res = self.avg_pool(x)
        b, s, m = pool_res.shape
        pool_res = torch_npu.npu_confusion_transpose(pool_res, (0, 1, 4, 2, 3),
                                                     (n, c, h, w, (b * s * m) // (n * c * h * w)), False)
        return alpha * pool_res + (1 - alpha) * self.conv(x)


class TimeUpsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        padding: int = 1,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size, padding)
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        if x.size(2) > 1:
            x, y = x[:, :, :1], x[:, :, 1:]
            x, y = x[:, :, :1], x[:, :, 1:]
            y = F.interpolate(y.float(), scale_factor=(2, 1, 1), mode="trilinear")
            x = torch.concat([x, y], dim=2)
        return alpha * x + (1 - alpha) * self.conv(x)


class Spatial2xTime2x3DDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="CausalConv3d"):
        super().__init__()
        if conv_type == "WfCausalConv3d":
            ConvLayer = WfCausalConv3d
        elif conv_type == "CausalConv3d":
            ConvLayer = CausalConv3d
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, padding=0, stride=2)

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Spatial2xTime2x3DUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if x.size(2) > 1:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            x_ = F.interpolate(x_, scale_factor=(2, 2, 2), mode="trilinear")
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
            x = torch.concat([x, x_], dim=2)
        else:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
        return self.conv(x)


class CachedCausal3DUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_interpolation="trilinear",
        enable_cached=False,
        depth=0,
    ):
        super().__init__()
        self.t_interpolation = t_interpolation
        self.conv = WfCausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.enable_cached = enable_cached
        self.causal_cached = deque()
        self.depth = depth

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)

        if x.size(2) == 1:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
            
        elif self.enable_cached:
            drop_cached = False
            if len(self.causal_cached) > 0:
                cached = self.causal_cached.popleft()
                x = torch.cat([cached, x], dim=2)
                drop_cached = True
            self.causal_cached.append(
                x[:, :, -(2**self.depth) - 1 : -(2**self.depth)].clone()
            )
            x = F.interpolate(x, scale_factor=(2, 1, 1), mode=self.t_interpolation)
            if drop_cached:
                x = x[:, :, 2:]
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
        else:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            x_ = F.interpolate(
                x_, scale_factor=(2, 1, 1), mode=self.t_interpolation
            )
            x_ = F.interpolate(x_, scale_factor=(1, 2, 2), mode="trilinear")
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
            x = torch.concat([x, x_], dim=2)
            
        x = x.to(x_dtype)
        return self.conv(x)