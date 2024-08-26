try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None

import torch
from .normalize import Normalize
from .ops import nonlinearity, video_to_image
from .conv import CausalConv3d
from .block import Block
from torch.utils.checkpoint import checkpoint


class ResnetBlock2D(Block):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        norm_type,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    @video_to_image
    def forward(self, x):
        h = x
        if npu_config is None:
            h = self.norm1(h)
        else:
            h = npu_config.run_group_norm(self.norm1, h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if npu_config is None:
            h = self.norm2(h)
        else:
            h = npu_config.run_group_norm(self.norm2, h)
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


class ResnetBlock3D(Block):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        norm_type,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(
                    in_channels, out_channels, 3, padding=1
                )
            else:
                self.nin_shortcut = CausalConv3d(
                    in_channels, out_channels, 1, padding=0
                )

    def forward(self, x):
        h = x
        if npu_config is None:
            h = self.norm1(h)
        else:
            h = npu_config.run_group_norm(self.norm1, h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if npu_config is None:
            h = self.norm2(h)
        else:
            h = npu_config.run_group_norm(self.norm2, h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class ResnetBlock3D_GC(Block):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        norm_type,
        dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type=norm_type)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(
                    in_channels, out_channels, 3, padding=1
                )
            else:
                self.nin_shortcut = CausalConv3d(
                    in_channels, out_channels, 1, padding=0
                )

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=True)

    def _forward(self, x):
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
