import random
import torch
from torch import nn
import torch.nn.functional as F


class PositionGetter3D:
    """return positions of patches"""

    def __init__(self, max_t, max_h, max_w, explicit_uniform_rope=False, atten_layout="BSH"):
        self.cache_positions = {}
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.explicit_uniform_rope = explicit_uniform_rope
        self.atten_layout = atten_layout

    def check_type(self, param):
        if isinstance(param, torch.Tensor):
            param = param.item()
        return param

    def __call__(self, b, t, h, w, device, training=True):
        b = self.check_type(b)
        t = self.check_type(t)
        h = self.check_type(h)
        w = self.check_type(w)

        # random.randint is [a, b], but torch.randint is [a, b)
        s_t = random.randint(0, self.max_t-t) if self.explicit_uniform_rope and training else 0
        e_t = s_t + t
        s_h = random.randint(0, self.max_h-h) if self.explicit_uniform_rope and training else 0
        e_h = s_h + h
        s_w = random.randint(0, self.max_w-w) if self.explicit_uniform_rope and training else 0
        e_w = s_w + w

        if not (b,s_t,e_t,s_h,e_h,s_w,e_w) in self.cache_positions:
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
            max_poses = (e_t, e_h, e_w)
            min_poses = (s_t, s_h, s_w)

            self.cache_positions[b,s_t,e_t,s_h,e_h,s_w,e_w] = (poses, min_poses, max_poses)
        pos = self.cache_positions[b,s_t,e_t,s_h,e_h,s_w,e_w]
        return pos


class RoPE3D(torch.nn.Module):

    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_start, seq_end, device, dtype, interpolation_scale=1):
        if (D, seq_start, seq_start, seq_end, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_start, seq_end, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_start, seq_start, seq_end, device, dtype] = (cos, sin)
        return self.cache[D, seq_start, seq_start, seq_end, device, dtype]

    def forward(self, dim, positions, device, dtype):
        """
        input:
            * dim: head_dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (ntokens x batch_size x nheads x dim)
        """
        assert dim % 16 == 0, "number of dimensions should be a multiple of 16"
        D_t = dim // 16 * 4
        D = dim // 16 * 6
        poses, min_poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2 # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D_t, min_poses[0], max_poses[0], device, dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, min_poses[1], max_poses[1], device, dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, min_poses[2], max_poses[2], device, dtype, self.interpolation_scale_w)

        cos_t, sin_t = compute_rope1d(poses[0], cos_t, sin_t)
        cos_y, sin_y = compute_rope1d(poses[1], cos_y, sin_y)
        cos_x, sin_x = compute_rope1d(poses[2], cos_x, sin_x)
        return cos_t, sin_t, cos_y, sin_y, cos_x, sin_x

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
    
def compute_rope1d(pos1d, cos, sin):
    """
        * pos1d: ntokens x batch_size
    """
    assert pos1d.ndim == 2
    # for (ntokens x batch_size x nheads x dim)
    cos = F.embedding(pos1d, cos)[:, :, None, :]
    sin = F.embedding(pos1d, sin)[:, :, None, :]

    return cos, sin

def apply_rope1d(tokens, cos, sin):
    """
        * tokens: ntokens x batch_size x nheads x dim
    """
    return (tokens * cos) + (rotate_half(tokens) * sin)
    
def apply_rotary_emb(tokens, video_rotary_emb):
    cos_t, sin_t, cos_y, sin_y, cos_x, sin_x = video_rotary_emb
    # split features into three along the feature dimension, and apply rope1d on each half
    dim = tokens.shape[-1]
    D_t = dim // 16 * 4
    D = dim // 16 * 6
    t, y, x = torch.split(tokens, [D_t, D, D], dim=-1)
    t = apply_rope1d(t, cos_t, sin_t)
    y = apply_rope1d(y, cos_y, sin_y)
    x = apply_rope1d(x, cos_x, sin_x)
    tokens = torch.cat((t, y, x), dim=-1)
    return tokens
    