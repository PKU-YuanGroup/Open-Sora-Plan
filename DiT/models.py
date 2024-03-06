# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from typing import Final, Optional

import torch
import torch.nn as nn
import numpy as np
import math
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.layers import use_fused_attn, to_2tuple
# from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.vision_transformer import PatchEmbed, Mlp
from torch.nn import functional as F


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attention_mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # import ipdb
            # ipdb.set_trace()
            # if attention_mask is not None:
            #     attention_mask = attention_mask.flatten(1).unsqueeze(-1)  # bs t h w -> bs thw 1
            #     attention_mask = attention_mask @ attention_mask.transpose(1, 2)  # bs thw 1 @ bs 1 thw = bs thw thw
            #     attention_mask = attention_mask.unsqueeze(1)
            #     assert attn.shape[0] % attention_mask.shape[0] == 0
            #     attention_mask = torch.cat([attention_mask] * (attn.shape[0] // attention_mask.shape[0]), dim=0)
            #     attention_mask = attention_mask.masked_fill(attention_mask == 0, torch.finfo(attn.dtype).min)
            #     attn = attn + attention_mask
            attn = attn.softmax(dim=-1)
            if torch.any(torch.isnan(attn)):
                print('torch.any(torch.isnan(attn))')
                attn = attn.masked_fill(torch.isnan(attn), float(0.))
            # import ipdb
            # ipdb.set_trace()
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conDiTioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attention_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size_t, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size_t * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        patch_size_t=1,
        in_channels=256,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.gradient_checkpointing = False

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # import ipdb;
        # ipdb.set_trace()
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size_t, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

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
        x: (B, N, patch_size_t * patch_size**2 * C)
        imgs: (B, C, T, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        o = self.patch_size_t
        assert self.t * self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.t, self.h, self.w, o, p, p, c))
        x = torch.einsum('nthwopqc->nctohpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.t * o, self.h * p, self.w * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y, attention_mask):
        """
        Forward pass of DiT.
        x: (B, D, T, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """
        if x.ndim == 4:
            raise NotImplementedError
        B, D, T, H, W = x.shape
        self.h = num_patches_height = H // self.patch_size
        self.w = num_patches_width = W // self.patch_size
        self.t = num_tubes_length = T // self.patch_size_t  # 4 // 1

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, [num_patches_height, num_patches_width])
        pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False).to(x.device)

        pos_embed_1d = get_2d_sincos_pos_embed(self.hidden_size, [num_tubes_length, 1])
        pos_embed_1d = nn.Parameter(torch.from_numpy(pos_embed_1d).float().unsqueeze(0), requires_grad=False).to(x.device)

        # print(num_patches_height, num_patches_width, x.shape, pos_embed.shape)
        self.x_embedder.img_size = [H, W]
        x = rearrange(x, 'b d t h w -> (b t) d h w')
        x = self.x_embedder(x) + pos_embed  # (BT, N, D), where N = H * W / patch_size ** 2
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.t)
        x = x + pos_embed_1d
        x = rearrange(x, '(b n) t d -> b (t n) d', b=B)

        t = self.t_embedder(t)                   # (B, D)
        y = self.y_embedder(y, self.training)    # (B, D)
        c = t + y                                # (B, D)
        for block in self.blocks:                  # (B, N, D)
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, attention_mask)  # (B, N, D)
            else:
                x = block(x, c, attention_mask)
        x = self.final_layer(x, c)                # (B, N, patch_size_t * patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, T, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, attention_mask):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, attention_mask)
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
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_122(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_144(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_188(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def DiT_L_122(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def DiT_L_144(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def DiT_L_188(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def DiT_B_122(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size_t=1, patch_size=2, num_heads=12, **kwargs)

def DiT_B_144(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size_t=1, patch_size=4, num_heads=12, **kwargs)

def DiT_B_188(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size_t=1, patch_size=8, num_heads=12, **kwargs)

def DiT_S_122(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size_t=1, patch_size=2, num_heads=6, **kwargs)

def DiT_S_144(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size_t=1, patch_size=4, num_heads=6, **kwargs)

def DiT_S_188(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size_t=1, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/122': DiT_XL_122,  'DiT-XL/144': DiT_XL_144,  'DiT-XL/188': DiT_XL_188,
    'DiT-L/122':  DiT_L_122,   'DiT-L/144':  DiT_L_144,   'DiT-L/188':  DiT_L_188,
    'DiT-B/122':  DiT_B_122,   'DiT-B/144':  DiT_B_144,   'DiT-B/188':  DiT_B_188,
    'DiT-S/122':  DiT_S_122,   'DiT-S/144':  DiT_S_144,   'DiT-S/188':  DiT_S_188,
}
