import torch.nn as nn
from typing import Union, Tuple
import torch.nn.functional as F
import torch
from .block import Block
from .ops import cast_tuple
from einops import rearrange
from .ops import video_to_image

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
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], init_method="random", **kwargs
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        stride = kwargs.pop("stride", 1)
        padding = kwargs.pop("padding", 0)
        padding = list(cast_tuple(padding, 3))
        padding[0] = 0
        stride = cast_tuple(stride, 3)
        self.conv = nn.Conv3d(chan_in, chan_out, self.kernel_size, stride=stride, padding=padding)
        self._init_weights(init_method)
        
    def _init_weights(self, init_method):
        ks = torch.tensor(self.kernel_size)
        if init_method == "avg":
            assert (
                self.kernel_size[1] == 1 and self.kernel_size[2] == 1
            ), "only support temporal up/down sample"
            assert self.chan_in == self.chan_out, "chan_in must be equal to chan_out"
            weight = torch.zeros((self.chan_out, self.chan_in, *self.kernel_size))

            eyes = torch.concat(
                [
                    torch.eye(self.chan_in).unsqueeze(-1) * 1/3,
                    torch.eye(self.chan_in).unsqueeze(-1) * 1/3,
                    torch.eye(self.chan_in).unsqueeze(-1) * 1/3,
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
                torch.zeros((self.chan_out, self.chan_in, *self.kernel_size)),
                requires_grad=True,
            )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
            
    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )   # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return self.conv(x)