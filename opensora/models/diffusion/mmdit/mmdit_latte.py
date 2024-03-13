# Modified from https://github.com/Vchitect/Latte


import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat

from opensora.models.diffusion.common.embed.clip_and_t5_text_emb import MixedTextEmbedder
from opensora.models.diffusion.common.embed.patch_emb import PatchEmbed3D
from opensora.models.diffusion.common.embed.pos_emb import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from opensora.models.diffusion.common.embed.time_emb import TimestepEmbedder
from opensora.models.diffusion.common.modules.mmdit_block import MMDiTBlock, MMDiTFinalLayer


class MMDiTLatte(nn.Module):
    """
    A MMDiT-like Latte.
    We only change the forward method in this class.
    The rest of the code aligns with MMDiT.
    """

    def __init__(
        self,
        clip_in_text_channels=384,
        clip_seq_len=77,
        t5_in_text_channels=4096,
        t5_seq_len=77,
        input_size=(16, 32, 32),
        patch_size=(1, 2, 2),
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        clip_text_encoder: str = None,
        t5_text_encoder: str = None,
        attention_mode='rebased',
        class_dropout_prob=0.1,
        learn_sigma=True,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        if enable_flashattn:
            assert dtype in [
                torch.float16,
                torch.bfloat16,
            ], f"Flash attention only supports float16 and bfloat16, but got {self.dtype}"

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = MixedTextEmbedder(clip_text_encoder, t5_text_encoder, hidden_size, dropout_prob=class_dropout_prob)

        self.register_buffer("pos_embed_spatial", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(hidden_size, hidden_size, hidden_size, num_heads, enable_layernorm_kernel or enable_modulate_kernel, mlp_ratio, attention_mode=attention_mode)
                for _ in range(depth)
            ]
        )
        self.final_layer = MMDiTFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)
        self.initialize_weights()

        self.gradient_checkpointing = False
        self.use_flash_attention = False

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

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad_:
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

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
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
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
        x: (B, C, T, H, W) tensor of inputs
        t: (B,) tensor of diffusion timesteps
        c: list of text
        """
        # origin inputs should be float32, cast to specified dtype
        x = x.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # (B, N, D)

        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial

        x = rearrange(x, "b t s d -> b (t s) d")

        t = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        с, ts_proj = self.с_embedder(с, self.training)  # (N, D)
        t = t + ts_proj
        condition = c

        condition_spatial = repeat(condition, "b d -> (b t) d", t=self.num_temporal)
        condition_temporal = repeat(condition, "b d -> (b s) d", s=self.num_spatial)

        # blocks
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                # spatial
                x = rearrange(x, "b (t s) d -> (b t) s d", t=self.num_temporal, s=self.num_spatial)
                c = condition_spatial
            else:
                # temporal
                x = rearrange(x, "b (t s) d -> (b s) t d", t=self.num_temporal, s=self.num_spatial)
                if i == 1:
                    x = x + self.pos_embed_temporal
                c = condition_temporal

            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.create_custom_forward(block), t, x, c)
            else:
                x = block(t, x, c)  # (B, N, D)

            if i % 2 == 0:
                x = rearrange(x, "(b t) s d -> b (t s) d", t=self.num_temporal, s=self.num_spatial)
            else:
                x = rearrange(x, "(b s) t d -> b (t s) d", t=self.num_temporal, s=self.num_spatial)

        # final process
        x = self.final_layer(x, condition)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)

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


#################################################################################
#                                   Model Configs                                  #
#################################################################################


def MMLatte_XL_2x2x2(**kwargs):
    return MMDiTLatte(
        depth=28,
        hidden_size=1152,
        patch_size=(2, 2, 2),
        num_heads=16,
        **kwargs,
    )


def MMLatte_XL_1x2x2(**kwargs):
    return MMDiTLatte(
        depth=28,
        hidden_size=1152,
        patch_size=(1, 2, 2),
        num_heads=16,
        **kwargs,
    )


MMLatte_models = {
    "MMLatte-XL/1x2x2": MMLatte_XL_1x2x2,
    "MMLatte-XL/2x2x2": MMLatte_XL_2x2x2,
}
