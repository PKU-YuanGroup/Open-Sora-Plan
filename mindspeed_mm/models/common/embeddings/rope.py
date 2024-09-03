import torch
from torch import nn
import torch.nn.functional as F
from megatron.core import mpu


class PositionGetter3D:
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, t, h, w, device):
        if not (b, t, h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            if mpu.get_context_parallel_world_size() > 1:
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            else:
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]
        return pos


class RoPE3D(nn.Module):

    def __init__(self, freq=10000.0, interpolation_scale=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.interpolation_scale_t, self.interpolation_scale_h, self.interpolation_scale_w = interpolation_scale
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

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        if pos1d.ndim != 2:
            raise AssertionError("pos1d.ndim must be 2")
        # for (batch_size x, ntokens x, nheads x, dim)
        cos = F.embedding(pos1d, cos)[:, :, None, :]
        sin = F.embedding(pos1d, sin)[:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            tokens: batch_size x nheads x ntokens x dim
            positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            tokens after appplying RoPE3D (batch_size x nheads x ntokens x x dim)
        """
        if tokens.size(3) % 3 != 0:
            raise AssertionError("number of dimensions should be a multiple of three")
        dim = tokens.size(3) // 3
        poses, max_poses = positions
        if len(poses) != 3 or poses[0].ndim != 2:  # [Batch, Seq, 3]
            raise AssertionError("poses shape error")
        cos_t, sin_t = self.get_cos_sin(dim, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(dim, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(dim, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens
