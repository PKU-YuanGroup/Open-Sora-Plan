try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    
import torch.nn as nn
from typing import Union, Tuple
import torch
from .block import Block
from .ops import cast_tuple
from .ops import video_to_image
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import deque

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



class CausalConv3d(Block):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        **kwargs
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
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
                if self.cache_offset == 0:
                    self.causal_cached.append(x[:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
                else:
                    self.causal_cached.append(x[:, :, :-self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
            else:
                self.causal_cached.append(x[:, :, 0:0, :, :].clone())
        elif self.enable_cached:
            self.causal_cached.append(x[:, :, 0:0, :, :].clone())
            
        x = self.conv(x)
        return x


class CausalConv3d_GC(CausalConv3d):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int]],
        init_method="random",
        **kwargs
    ):
        super().__init__(chan_in, chan_out, kernel_size, init_method, **kwargs)

    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )  # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return checkpoint(self.conv, x)
