# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# croco: https://github.com/naver/croco
# diffusers: https://github.com/huggingface/diffusers
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import torch


def get_2d_sincos_pos_embed(
        embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(
        embed_dim, length, interpolation_scale=1.0, base_size=16
):
    pos = torch.arange(0, length).unsqueeze(1) / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# ----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
# ----------------------------------------------------------

try:
    from .curope import cuRoPE2D

    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')


    class RoPE2D(torch.nn.Module):

        def __init__(self, freq=10000.0, F0=1.0, scaling_factor=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.scaling_factor = scaling_factor
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D, seq_len, device, dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D, seq_len, device, dtype] = (cos, sin)
            return self.cache[D, seq_len, device, dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim == 2
            
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
            x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens

class LinearScalingRoPE2D(RoPE2D):
    """Code from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L148"""

    def forward(self, tokens, positions):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        dtype = positions.dtype
        positions = positions.float() / self.scaling_factor
        positions = positions.to(dtype)
        tokens = super().forward(tokens, positions)
        return tokens


try:
    from .curope import cuRoPE1D

    RoPE1D = cuRoPE1D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')


    class RoPE1D(torch.nn.Module):

        def __init__(self, freq=10000.0, F0=1.0, scaling_factor=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.scaling_factor = scaling_factor
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D, seq_len, device, dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D, seq_len, device, dtype] = (cos, sin)
            return self.cache[D, seq_len, device, dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim == 2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens (t position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            D = tokens.size(3)
            assert positions.ndim == 2  # Batch, Seq
            cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
            tokens = self.apply_rope1d(tokens, positions, cos, sin)
            return tokens

class LinearScalingRoPE1D(RoPE1D):
    """Code from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L148"""
    
    def forward(self, tokens, positions):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        dtype = positions.dtype
        positions = positions.float() / self.scaling_factor
        positions = positions.to(dtype)
        tokens = super().forward(tokens, positions)
        return tokens


class PositionGetter2D(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos



class PositionGetter1D(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, l, device):
        if not (l) in self.cache_positions:
            x = torch.arange(l, device=device)
            self.cache_positions[l] = x # (l, )
        pos = self.cache_positions[l].view(1, l).expand(b, -1).clone()
        return pos