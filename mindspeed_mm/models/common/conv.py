from typing import Union, Tuple
from collections import deque

import torch
from torch import nn
import torch_npu

from mindspeed_mm.utils.utils import cast_tuple, video_to_image


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    @video_to_image
    def forward(self, x):
        return super().forward(x)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        init_method: str = "random",
        **kwargs
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = kwargs.pop("stride", 1)
        padding = kwargs.pop("padding", 0)
        padding = list(cast_tuple(padding, 3))
        padding[0] = 0
        stride = cast_tuple(stride, 3)
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size, stride=stride, padding=padding)
        self.pad = nn.ReplicationPad2d((0, 0, self.time_kernel_size - 1, 0))
        if init_method:
            self._init_weights(init_method)

    def _init_weights(self, init_method):
        if init_method == "avg":
            if not (self.kernel_size[1] == 1 and self.kernel_size[2] == 1):
                raise AssertionError("only support temporal up/down sample")
            if self.in_channels != self.out_channels:
                raise AssertionError("in_channels must be equal to out_channels")
            weight = torch.zeros((self.out_channels, self.in_channels, *self.kernel_size))

            eyes = torch.concat(
                [
                    torch.eye(self.in_channels).unsqueeze(-1) * 1 / 3,
                    torch.eye(self.in_channels).unsqueeze(-1) * 1 / 3,
                    torch.eye(self.in_channels).unsqueeze(-1) * 1 / 3,
                ],
                dim=-1,
            )
            weight[:, :, :, 0, 0] = eyes

            self.conv.weight = nn.Parameter(
                weight,
                requires_grad=True,
            )
        elif init_method == "zero":
            self.conv.weight = nn.Parameter(
                torch.zeros((self.out_channels, self.in_channels, *self.kernel_size)),
                requires_grad=True,
            )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))  # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return self.conv(x)


class WfCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias
        )
        self.enable_cached = enable_cached

        self.is_first_chunk = True

        self.causal_cached = deque()
        self.cache_offset = 0

    def forward(self, x):
        if self.is_first_chunk:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
        else:
            first_frame_pad = self.causal_cached.popleft()
            
        x = torch.concatenate((first_frame_pad, x), dim=2)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0 or self.is_first_chunk:
                    self.causal_cached.append(x[:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
                else:
                    self.causal_cached.append(x[:, :, :-self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
            else:
                self.causal_cached.append(x[:, :, 0:0, :, :].clone())
        elif self.enable_cached:
            self.causal_cached.append(x[:, :, 0:0, :, :].clone())
            
        # if x.dtype not in [torch.float16, torch.bfloat16]:
        #     dtype = x.dtype
        #     with torch.cuda.amp.autocast(enabled=False):
        #         x = self.conv.to(device=x.device, dtype=torch.bfloat16)(x.to(torch.bfloat16))
        #         x = x.to(dtype)
        #         return torch_npu.npu_format_cast(x, 2)
        # else:
        return self.conv(x)
