import torch
from torch import nn
import torch.nn.functional as F


class PositionGetter3D:
    """return positions of patches"""

    def __init__(self, atten_layout="BSH"):
        self.cache_positions = {}
        self.atten_layout = atten_layout

    def check_type(self, param):
        if isinstance(param, torch.Tensor):
            param = param.item()
        return param

    def __call__(self, b, t, h, w, device):
        b = self.check_type(b)
        t = self.check_type(t)
        h = self.check_type(h)
        w = self.check_type(w)

        if not (b, t, h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            if self.atten_layout == "SBH":
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            elif self.atten_layout == "BSH":
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
            else:
                raise ValueError(f"Unsupported layout type: {self.atten_layout}")
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]
        return pos


class RoPE3D(nn.Module):

    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, dim, seq_len, device, dtype, interpolation_scale=1):
        if (dim, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float().to(device) / dim))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[dim, seq_len, device, dtype] = (cos, sin)
        return self.cache[dim, seq_len, device, dtype]

    def forward(self, dim, positions, device, dtype):
        """
        input:
            * dim: head_dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (ntokens x batch_size x nheads x dim)
        """
        assert dim % 16 == 0, "number of dimensions should be a multiple of 16"
        dim_t = dim // 16 * 4
        dim_hw = dim // 16 * 6
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2 # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(dim_t, max_poses[0] + 1, device, dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(dim_hw, max_poses[1] + 1, device, dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(dim_hw, max_poses[2] + 1, device, dtype, self.interpolation_scale_w)
        return poses, cos_t, sin_t, cos_y, sin_y, cos_x, sin_x

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope1d(tokens, pos1d, cos, sin):
    """
        * tokens: ntokens x batch_size x nheads x dim
        * pos1d: ntokens x batch_size
    """
    assert pos1d.ndim == 2
    # for (ntokens x batch_size x nheads x dim)
    cos = F.embedding(pos1d, cos)[:, :, None, :]
    sin = F.embedding(pos1d, sin)[:, :, None, :]

    return (tokens * cos) + (rotate_half(tokens) * sin)

def apply_rotary_emb(tokens, video_rotary_emb):
    poses, cos_t, sin_t, cos_y, sin_y, cos_x, sin_x = video_rotary_emb
    # split features into three along the feature dimension, and apply rope1d on each half
    dim = tokens.shape[-1]
    dim_t = dim // 16 * 4
    dim_hw = dim // 16 * 6
    t, y, x = torch.split(tokens, [dim_t, dim_hw, dim_hw], dim=-1)
    t = apply_rope1d(t, poses[0], cos_t, sin_t)
    y = apply_rope1d(y, poses[1], cos_y, sin_y)
    x = apply_rope1d(x, poses[2], cos_x, sin_x)
    tokens = torch.cat((t, y, x), dim=-1)
    return tokens
    