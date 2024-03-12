# Modified from Meta DiT and StabilityAI's StableDiffusion3 paper

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:   https://github.com/facebookresearch/DiT/tree/main
# GLIDE: https://github.com/openai/glide-text2im
# MAE:   https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed
from torch.distributed import ProcessGroup

from opensora.models.diffusion.common.embed.clip_and_t5_text_emb import MixedTextEmbedder
from opensora.models.diffusion.common.embed.patch_emb import PatchEmbed3D
from opensora.models.diffusion.common.embed.pos_emb import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from opensora.models.diffusion.common.embed.time_emb import TimestepEmbedder
from opensora.models.diffusion.common.modules.mmdit_block import MMDiTBlock, MMDiTFinalLayer
from opensora.models.diffusion.common.util.operation import gather_forward_split_backward


class MMDiT(nn.Module):
    """
    Diffusion model with a Multimodal Transformer backbone.
    NOTE: renamed y to c to match SD3's paper
    """

    def __init__(
        self,
        clip_in_text_channels=384,
        clip_seq_len=77,
        t5_in_text_channels=4096,
        t5_seq_len=77,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma: bool = True,
        enable_flashattn: bool = False,
        enable_layernorm_kernel: bool = False,
        enable_modulate_kernel: bool = False,
        sequence_parallel_size: int = 1,
        sequence_parallel_group: Optional[ProcessGroup] = None,
        sequence_parallel_type: str = None,
        dtype: torch.dtype = torch.float32,
        use_video: bool = False,
        clip_text_encoder: str = None,
        t5_text_encoder: str = None,
        attention_mode='rebased',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.use_video = use_video
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_group = sequence_parallel_group
        self.sequence_parallel_type = sequence_parallel_type

        self.dtype = dtype
        if enable_flashattn:
            assert dtype in [
                torch.float16,
                torch.bfloat16,
            ], f"Flash attention only supports float16 and bfloat16, but got {self.dtype}"

        # For img input, use PatchEmbed and LabelEmbedder
        # For video input, use PatchEmbed3D and TextEmbedder
        if self.use_video:
            self.x_embedder = PatchEmbed3D(patch_size, in_channels, embed_dim=hidden_size)
            self.t_embedder = TimestepEmbedder(hidden_size)
            self.c_embedder = MixedTextEmbedder(clip_text_encoder, t5_text_encoder, hidden_size)
            self.num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        else:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            self.t_embedder = TimestepEmbedder(hidden_size)
            self.c_embedder = MixedTextEmbedder(clip_text_encoder, t5_text_encoder, hidden_size)
            self.num_patches = self.x_embedder.num_patches

        if self.use_video:
            self.num_temporal = input_size[0] // patch_size[0]
            self.num_spatial = self.num_patches // self.num_temporal
            self.register_buffer("pos_embed_spatial", self.get_spatial_pos_embed())
            self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(hidden_size, hidden_size, hidden_size, num_heads, enable_layernorm_kernel or enable_modulate_kernel, mlp_ratio, attention_mode=attention_mode)
                for _ in range(depth)
            ]
        )
        self.final_layer = MMDiTFinalLayer(
            hidden_size, np.prod(patch_size) if use_video else patch_size ** 2, self.out_channels
        )
        self.initialize_weights()

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def get_spatial_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[1] // self.patch_size[1],
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if not self.use_video:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def unpatchify3D(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = torch.einsum("nthwrpqc->nctrhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    def forward(self, x, t, c):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        c: (N,) texts -> clip embeddings tensor
        """

        # origin inputs should be float32, cast to specified dtype
        x = x.to(self.dtype)

        if self.use_video:
            x = self.x_embedder(x)
            x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
            x = x + self.pos_embed_spatial
            x = rearrange(x, "b t s d -> b s t d")
            x = x + self.pos_embed_temporal
            x = rearrange(x, "b s t d -> b (t s) d")
        else:
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t, dtype=x.dtype)  # (N, D)

        c, ts_proj = self.c_embedder(c, self.training)
        text_embedding_spatial = repeat(c, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        text_embedding_temp = repeat(c, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        t = t + ts_proj

        # Chunk x on sequence dimension to sp group
        if self.sequence_parallel_size > 1:
            x = x.chunk(self.sequence_parallel_size, dim=1)[dist.get_rank(self.sequence_parallel_group)]

        for block in self.blocks:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.create_custom_forward(block), t, x, c)
            else:
                x = block(t, x, c)  # (N, T, D)

        if self.sequence_parallel_size > 1:
            x = gather_forward_split_backward(x, dim=1, process_group=self.sequence_parallel_group)

        c = t
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        if self.use_video:
            x = self.unpatchify3D(x)
        else:
            x = self.unpatchify(x)  # (N, out_channels, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def rearrange_attention_weights(self, flag="load"):
        for block in self.blocks:
            block.attn.rearrange_fused_weight(block.attn.qkv, flag)
        torch.cuda.empty_cache()


#################################################################################
#                                 MMDiT Configs                                  #
#################################################################################


def MMDiT_XL_2(**kwargs):
    return MMDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def MMDiT_XL_4(**kwargs):
    return MMDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def MMDiT_XL_8(**kwargs):
    return MMDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def MMDiT_L_2(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def MMDiT_L_4(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def MMDiT_L_8(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def MMDiT_B_2(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def MMDiT_B_4(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def MMDiT_B_8(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def MMDiT_S_2(**kwargs):
    return MMDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def MMDiT_S_4(**kwargs):
    return MMDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def MMDiT_S_8(**kwargs):
    return MMDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


def MMVDiT_XL_1x2x2(**kwargs):
    return MMDiT(
        depth=28,
        hidden_size=1152,
        patch_size=(1, 2, 2),
        num_heads=16,
        use_video=True,
        **kwargs,
    )


def MMVDiT_XL_2x2x2(**kwargs):
    return MMDiT(
        depth=28,
        hidden_size=1152,
        patch_size=(2, 2, 2),
        num_heads=16,
        use_video=True,
        **kwargs,
    )


MMDiT_models = {
    # image model
    "MMDiT-XL/2": MMDiT_XL_2,
    "MMDiT-XL/4": MMDiT_XL_4,
    "MMDiT-XL/8": MMDiT_XL_8,
    "MMDiT-L/2": MMDiT_L_2,
    "MMDiT-L/4": MMDiT_L_4,
    "MMDiT-L/8": MMDiT_L_8,
    "MMDiT-B/2": MMDiT_B_2,
    "MMDiT-B/4": MMDiT_B_4,
    "MMDiT-B/8": MMDiT_B_8,
    "MMDiT-S/2": MMDiT_S_2,
    "MMDiT-S/4": MMDiT_S_4,
    "MMDiT-S/8": MMDiT_S_8,
    # video model
    "MMVDiT-XL/1x2x2": MMVDiT_XL_1x2x2,
    "MMVDiT-XL/2x2x2": MMVDiT_XL_2x2x2,
}
