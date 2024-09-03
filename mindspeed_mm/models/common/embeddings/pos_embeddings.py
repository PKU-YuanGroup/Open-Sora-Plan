# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2024 HPC-AI Technology Inc. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=(1.0, 1.0, 1.0),
    base_size=None,
) -> np.array:
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid height and width
    return:
        pos_embed: [grid_size*grid_size, embed_dim] or
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_t = np.arange(grid_size[0], dtype=np.float32) / interpolation_scale[0]
    grid_h = np.arange(grid_size[1], dtype=np.float32) / interpolation_scale[1]
    grid_w = np.arange(grid_size[2], dtype=np.float32) / interpolation_scale[2]
    if base_size is not None:
        grid_t *= base_size / grid_size[0]
        grid_h *= base_size / grid_size[1]
        grid_w *= base_size / grid_size[2]
    grid = np.meshgrid(grid_w, grid_h, grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[2], grid_size[1], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid) -> np.array:
    """
    embed_dim: output dimension for each position
    grid: list of grid size
    """
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (T*H*W, D/3)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (T*H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=(1.0, 1.0),
    base_size=None,
) -> np.array:
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid height and width
    return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32) / interpolation_scale[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / interpolation_scale[1]
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid) -> np.array:
    """
    embed_dim: output dimension for each position
    grid: list of grid size
    """

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos) -> np.array:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=None,
) -> np.array:
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid
    return:
        pos_embed: [grid_size, embed_dim] or
        [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid = np.arange(grid_size, dtype=np.float32) / interpolation_scale
    if base_size is not None:
        grid *= base_size / grid_size

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed
