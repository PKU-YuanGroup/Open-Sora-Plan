# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# EVA: https://github.com/baaivision/EVA
# Transformer升级之路：2、博采众长的旋转式位置编码: https://spaces.ac.cn/archives/8265
# Transformer升级之路：4、二维位置的旋转式位置编码: https://spaces.ac.cn/archives/8397
# --------------------------------------------------------
import torch
import torch.nn as nn
from math import pi
from einops import rearrange, repeat


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_hw=(int, int),  # (H, W)
        ft_hw=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        # Unlike a 1d RoPE, a 2d RoPE requires splitting the dimension into four parts
        # References: https://spaces.ac.cn/archives/8397

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_hw is None: ft_hw = pt_hw
        h_t = torch.arange(ft_hw[0]) / ft_hw[0] * pt_hw[0]
        w_t = torch.arange(ft_hw[1]) / ft_hw[1] * pt_hw[1]

        h_freqs = torch.einsum('..., f -> ... f', h_t, freqs)
        w_freqs = torch.einsum('..., f -> ... f', w_t, freqs)

        h_freqs = repeat(h_freqs, '... n -> ... (n r)', r=2)
        w_freqs = repeat(w_freqs, '... n -> ... (n r)', r=2)

        freqs = broadcat((h_freqs[:, None, :], w_freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        # 2d RoPE: [[cos(h*theta), -sin(h*theta), 0,            0            ],
        #           [sin(h*theta), cos(h*theta),  0,            0            ],
        #           [0,            0,             cos(w*theta), -sin(w*theta)],
        #           [0,            0,             sin(w*theta), cos(w*theta) ],]

        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin