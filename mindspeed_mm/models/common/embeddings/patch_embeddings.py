# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from einops import rearrange
from megatron.training import get_args
from torch import nn

from .pos_embeddings import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)


class VideoPatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding but with 3D position embedding
    """

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
        eps=1e-6,
    ):
        super().__init__()
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            (self.height, self.width),
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
        )
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            persistent=False,
        )

        self.num_frames = (
            (num_frames - 1) // patch_size_t + 1
            if num_frames % 2 == 1
            else num_frames // patch_size_t
        )
        self.base_size_t = (
            (num_frames - 1) // patch_size_t + 1
            if num_frames % 2 == 1
            else num_frames // patch_size_t
        )
        self.interpolation_scale_t = interpolation_scale_t

        temp_pos_embed = get_1d_sincos_pos_embed(
            embed_dim,
            self.num_frames,
            base_size=self.base_size_t,
            interpolation_scale=self.interpolation_scale_t,
        )
        self.register_buffer(
            "temp_pos_embed",
            torch.from_numpy(temp_pos_embed).float().unsqueeze(0),
            persistent=False,
        )

        self.args = get_args()

    def forward(self, latent, num_frames):
        """
        batch, channel, time, height, width
        """
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        height, width = (
            latent.shape[-2] // self.patch_size,
            latent.shape[-1] // self.patch_size,
        )
        latent = rearrange(latent, "b c t h w -> (b t) c h w")
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                if self.args.sequence_parallel:
                    sp_size = self.args.world_size
                    temp_pos_embed = get_1d_sincos_pos_embed(
                        embed_dim=self.temp_pos_embed.shape[-1],
                        grid_size=num_frames * sp_size,
                        base_size=self.base_size_t,
                        interpolation_scale=self.interpolation_scale_t,
                    )
                    rank = self.args.rank % sp_size
                    st_frame = rank * num_frames
                    ed_frame = st_frame + num_frames
                    temp_pos_embed = temp_pos_embed[st_frame:ed_frame]

                else:
                    temp_pos_embed = get_1d_sincos_pos_embed(
                        embed_dim=self.temp_pos_embed.shape[-1],
                        grid_size=num_frames,
                        base_size=self.base_size_t,
                        interpolation_scale=self.interpolation_scale_t,
                    )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        latent = rearrange(latent, "(b t) n c -> b t n c", b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )

        video_latent = (
            rearrange(video_latent, "b t n c -> b (t n) c")
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        image_latent = (
            rearrange(image_latent, "b t n c -> (b t) n c")
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None and not self.args.sequence_parallel:
            image_latent = video_latent
            video_latent = None

        return video_latent, image_latent


class OverlapPatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
        eps=1e-6,
    ):
        super().__init__()
        if patch_size_t != 1:
            raise ValueError("patch_size_t must be 1 in OverlapPatchEmbed2D")
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            (self.height, self.width),
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
        )
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            persistent=False,
        )

        self.num_frames = (
            (num_frames - 1) // patch_size_t + 1
            if num_frames % 2 == 1
            else num_frames // patch_size_t
        )
        self.base_size_t = (
            (num_frames - 1) // patch_size_t + 1
            if num_frames % 2 == 1
            else num_frames // patch_size_t
        )
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(
            embed_dim,
            self.num_frames,
            base_size=self.base_size_t,
            interpolation_scale=self.interpolation_scale_t,
        )
        self.register_buffer(
            "temp_pos_embed",
            torch.from_numpy(temp_pos_embed).float().unsqueeze(0),
            persistent=False,
        )

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None

        # b c 1 h w
        height, width = (
            latent.shape[-2] // self.patch_size,
            latent.shape[-1] // self.patch_size,
        )
        latent = rearrange(latent, "b c t h w -> (b t) c h w")
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        latent = rearrange(latent, "(b t) n c -> b t n c", b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )

        video_latent = (
            rearrange(video_latent, "b t n c -> b (t n) c")
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        image_latent = (
            rearrange(image_latent, "b t n c -> (b t) n c")
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None
        return video_latent, image_latent


class OverlapPatchEmbed3D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1,
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=True,
        eps=1e-06,
    ):
        super().__init__()
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size_t, patch_size, patch_size),
            stride=(patch_size_t, patch_size, patch_size),
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=eps)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            (self.height, self.width),
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
        )
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            persistent=False,
        )

        self.num_frames = (
            (num_frames - 1) // patch_size_t + 1
            if num_frames % 2 == 1
            else num_frames // patch_size_t
        )
        self.base_size_t = (
            (num_frames - 1) // patch_size_t + 1
            if num_frames % 2 == 1
            else num_frames // patch_size_t
        )
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(
            embed_dim,
            self.num_frames,
            base_size=self.base_size_t,
            interpolation_scale=self.interpolation_scale_t,
        )
        self.register_buffer(
            "temp_pos_embed",
            torch.from_numpy(temp_pos_embed).float().unsqueeze(0),
            persistent=False,
        )

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        height, width = (
            latent.shape[-2] // self.patch_size,
            latent.shape[-1] // self.patch_size,
        )
        latent = self.proj(latent)

        if self.flatten:
            latent = rearrange(latent, "b c t h w -> (b t) (h w) c ")
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

            if self.num_frames != num_frames:
                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)

        latent = rearrange(latent, "(b t) n c -> b t n c", b=b)
        video_latent, image_latent = latent[:, :num_frames], latent[:, num_frames:]

        if self.use_abs_pos:
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (
                (video_latent + temp_pos_embed).to(video_latent.dtype)
                if video_latent is not None and video_latent.numel() > 0
                else None
            )
            image_latent = (
                (image_latent + temp_pos_embed[:, :1]).to(image_latent.dtype)
                if image_latent is not None and image_latent.numel() > 0
                else None
            )

        video_latent = (
            rearrange(video_latent, "b t n c -> b (t n) c")
            if video_latent is not None and video_latent.numel() > 0
            else None
        )
        image_latent = (
            rearrange(image_latent, "b t n c -> (b t) n c")
            if image_latent is not None and image_latent.numel() > 0
            else None
        )

        if num_frames == 1 and image_latent is None:
            image_latent = video_latent
            video_latent = None
        return video_latent, image_latent


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with video"""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)
        latent = rearrange(latent, '(b t) c h w -> b (t h w) c', b=b)
        return latent