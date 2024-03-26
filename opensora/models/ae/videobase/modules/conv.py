import torch.nn as nn
from typing import Union, Tuple
import torch.nn.functional as F
import torch
from einops import rearrange, pack, unpack
from .ops import cast_tuple

class CausalConv3d(nn.Module):
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], **kwargs
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
        self._init_weights()
        
    def _init_weights(self):
        ks = torch.tensor(self.kernel_size)
        mean_weight = torch.full((self.chan_out, self.chan_in, *self.kernel_size), fill_value=1 / (torch.prod(ks) * self.chan_in))
        self.conv.weight = nn.Parameter(mean_weight, requires_grad=True)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
            
    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return self.conv(x)
    