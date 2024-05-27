from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_block import ResnetBlock3D
from .attention import TemporalAttnBlock
from .normalize import Normalize
from .ops import cast_tuple, video_to_image
from .conv import CausalConv3d
from einops import rearrange
from .block import Block

class Upsample(Block):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            
    @video_to_image
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(Block):
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
    @video_to_image
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class SpatialDownsample2x(Block):
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


class SpatialUpsample2x(Block):
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
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> b (c t) h w")
        x = F.interpolate(x, scale_factor=(2,2), mode="nearest")
        x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = self.conv(x)
        return x
    
class TimeDownsample2x(Block):
    def __init__(
        self,
        chan_in,
        chan_out,
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

class TimeUpsample2x(Block):
    def __init__(
        self,
        chan_in,
        chan_out
    ):
        super().__init__()
    def forward(self, x):
        if x.size(2) > 1:
            x,x_= x[:,:,:1],x[:,:,1:]
            x_= F.interpolate(x_, scale_factor=(2,1,1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        return x
    
class TimeDownsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.avg_pool = nn.AvgPool3d((kernel_size,1,1), stride=(2,1,1))
        self.conv = nn.Conv3d(
            in_channels, out_channels, self.kernel_size, stride=(2,1,1), padding=(0,1,1)
        )
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))
    
    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.kernel_size[0] - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return alpha * self.avg_pool(x) + (1 - alpha) * self.conv(x)

class TimeUpsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 2.0,
    ):
        super().__init__()
        self.conv = CausalConv3d(
            in_channels, out_channels, kernel_size, padding=1
        )
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))
        
    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        if x.size(2) > 1:
            x,x_= x[:,:,:1],x[:,:,1:]
            x_= F.interpolate(x_, scale_factor=(2,1,1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        return alpha * x + (1-alpha) * self.conv(x)

class TimeDownsampleResAdv2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 1.5,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.avg_pool = nn.AvgPool3d((kernel_size,1,1), stride=(2,1,1))
        self.attn = TemporalAttnBlock(in_channels)
        self.res = ResnetBlock3D(in_channels=in_channels, out_channels=in_channels, dropout=0.0)
        self.conv = nn.Conv3d(
            in_channels, out_channels, self.kernel_size, stride=(2,1,1), padding=(0,1,1)
        )
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))
    
    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.kernel_size[0] - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        alpha = torch.sigmoid(self.mix_factor)
        return alpha * self.avg_pool(x) + (1 - alpha) * self.conv(self.attn((self.res(x))))

class TimeUpsampleResAdv2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        mix_factor: float = 1.5,
    ):
        super().__init__()
        self.res = ResnetBlock3D(in_channels=in_channels, out_channels=in_channels, dropout=0.0)
        self.attn = TemporalAttnBlock(in_channels)
        self.norm = Normalize(in_channels=in_channels)
        self.conv = CausalConv3d(
            in_channels, out_channels, kernel_size, padding=1
        )
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))
        
    def forward(self, x):
        if x.size(2) > 1:
            x,x_= x[:,:,:1],x[:,:,1:]
            x_= F.interpolate(x_, scale_factor=(2,1,1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        alpha = torch.sigmoid(self.mix_factor)
        return alpha * x + (1 - alpha) * self.conv(self.attn(self.res(x)))
