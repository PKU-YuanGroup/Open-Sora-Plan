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
        self.conv = nn.Conv3d(
            chan_in, chan_out, self.kernel_size, stride=stride, padding=padding
        )
        self._init_weights()

    def _init_weights(self):
        ks = torch.tensor(self.kernel_size)
        mean_weight = torch.full(
            (self.chan_out, self.chan_in, *self.kernel_size),
            fill_value=1 / (torch.prod(ks) * self.chan_in),
        )
        self.conv.weight = nn.Parameter(mean_weight, requires_grad=True)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )
        x = torch.concatenate((first_frame_pad, x), dim=2)
        return self.conv(x)


def cal_idx_by_step_begin(kernel_size, stride, step):
    return stride * step

def cal_idx_by_step_end(kernel_size, stride, step):
    return kernel_size + (step - 1) * stride

def get_max_step(kernel_size, stride, length):
    return (length - kernel_size) // stride + 1

def get_new_size(kernel_size, stride, length):
    return (length - kernel_size) // stride  + 1


class BlockwiseConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        block_step=(15, 15, 15),
    ):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.tile_padding = torch.repeat_interleave(torch.tensor(padding), 2).tolist()
        self.tile_padding.reverse()
        self.stride = stride
        self.block_step = block_step
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride, padding=0)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        dtype = x.dtype
        padded_x = nn.functional.pad(x, self.tile_padding)
        del x
        time_length = get_new_size(
            self.kernel_size[0], self.stride[0], padded_x.shape[2]
        )
        height_length = get_new_size(
            self.kernel_size[1], self.stride[1], padded_x.shape[3]
        )
        width_length = get_new_size(
            self.kernel_size[2], self.stride[2], padded_x.shape[4]
        )

        result = torch.zeros(
            (batch_size, self.chan_out, time_length, height_length, width_length), device=device, dtype=dtype
        )

        origin_time_length = padded_x.shape[2]
        origin_height_length = padded_x.shape[3]
        origin_width_length = padded_x.shape[4]

        for time in range(
            0,
            get_max_step(self.kernel_size[0], self.stride[0], origin_time_length),
            self.block_step[0],
        ):
            for height in range(
                0,
                get_max_step(self.kernel_size[1], self.stride[1], origin_height_length),
                self.block_step[1],
            ):
                for width in range(
                    0,
                    get_max_step(
                        self.kernel_size[2], self.stride[2], origin_width_length
                    ),
                    self.block_step[2],
                ):
                    time_start = cal_idx_by_step_begin(
                        self.kernel_size[0], self.stride[0], time
                    )
                    height_start = cal_idx_by_step_begin(
                        self.kernel_size[1], self.stride[1], height
                    )
                    width_start = cal_idx_by_step_begin(
                        self.kernel_size[2], self.stride[2], width
                    )
                    if (
                        time_start >= origin_time_length
                        or height_start >= origin_height_length
                        or width_start >= origin_width_length
                    ):
                        continue
                    time_end = cal_idx_by_step_end(
                        self.kernel_size[0], self.stride[0], time + self.block_step[0]
                    )
                    height_end = cal_idx_by_step_end(
                        self.kernel_size[1], self.stride[1], height + self.block_step[1]
                    )
                    width_end = cal_idx_by_step_end(
                        self.kernel_size[2], self.stride[2], width + self.block_step[2]
                    )
                    block = padded_x[
                        :,
                        :,
                        time_start : min(time_end, origin_time_length),
                        height_start : min(height_end, origin_height_length),
                        width_start : min(width_end, origin_width_length),
                    ]
                    result_block = self.conv(block)
                    result[
                        :,
                        :,
                        time : time + self.block_step[0],
                        height : height + self.block_step[1],
                        width : width + self.block_step[2],
                    ] = result_block
        return result

    def set_weights(self, weights, bias):
        self.conv.weight = nn.Parameter(weights)
        self.conv.bias = nn.Parameter(bias)