import torch

class PositionGetter3D(object):
    """ return positions of patches """

    def __init__(self, ):
        self.cache_positions = {}
        
    def __call__(self, b, t, h, w, device):
        if not (t,h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            self.cache_positions[t,h,w] = torch.cartesian_prod(z, y, x) # (t, h, w, 3)
        pos = self.cache_positions[t,h,w].view(1, t*h*w, 3).expand(b, -1, 3).clone()
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

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
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
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        assert positions.ndim == 3 and positions.shape[-1] == 3  # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D, int(positions[:, :, 0].max()) + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, int(positions[:, :, 1].max()) + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, int(positions[:, :, 2].max()) + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, positions[:, :, 0], cos_t, sin_t)
        y = self.apply_rope1d(y, positions[:, :, 1], cos_y, sin_y)
        x = self.apply_rope1d(x, positions[:, :, 2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens



# import torch
# from einops import rearrange, repeat

# class PositionGetter3D(object):
#     """ return positions of patches """

#     def __init__(self, ):
#         self.cache_positions = {}
        
#     def __call__(self, b, t, h, w, device):
#         if not (t,h,w) in self.cache_positions:
#             x = torch.arange(w, device=device)
#             y = torch.arange(h, device=device)
#             z = torch.arange(t, device=device)
#             positions = torch.cartesian_prod(z, y, x) # (t, h, w, 3)
#             positions = rearrange(positions, 'n d -> d 1 n')
#             positions = repeat(positions, 'd 1 n -> d b n', b=b).clone()
#             poses = (positions[0], positions[1], positions[2])
#             max_pos = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))
#             self.cache_positions[t,h,w] = (poses, max_pos)
#         pos = self.cache_positions[t,h,w]
#         return pos
    

# class RoPE3D(torch.nn.Module):

#     def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
#         super().__init__()
#         self.base = freq
#         self.F0 = F0
#         self.interpolation_scale_t = interpolation_scale_thw[0]
#         self.interpolation_scale_h = interpolation_scale_thw[1]
#         self.interpolation_scale_w = interpolation_scale_thw[2]
#         self.cache = {}

#     def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
#         if (D, seq_len, device, dtype) not in self.cache:
#             inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
#             t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
#             freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
#             freqs = torch.cat((freqs, freqs), dim=-1)
#             cos = freqs.cos()  # (Seq, Dim)
#             sin = freqs.sin()
#             self.cache[D, seq_len, device, dtype] = (cos, sin)
#         return self.cache[D, seq_len, device, dtype]

#     @staticmethod
#     def rotate_half(x):
#         x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
#         return torch.cat((-x2, x1), dim=-1)

#     def apply_rope1d(self, tokens, pos1d, cos, sin):
#         assert pos1d.ndim == 2
#         cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
#         sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
#         return (tokens * cos) + (self.rotate_half(tokens) * sin)

#     def forward(self, tokens, positions):
#         """
#         input:
#             * tokens: batch_size x nheads x ntokens x dim
#             * positions: batch_size x ntokens x 3 (t, y and x position of each token)
#         output:
#             * tokens after appplying RoPE3D (batch_size x nheads x ntokens x dim)
#         """
#         assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
#         D = tokens.size(3) // 3
#         poses, max_pos = positions
#         assert len(poses) == 3 and poses[0].ndim == 2   # Batch, Seq, 3
#         cos_t, sin_t = self.get_cos_sin(D, max_pos[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
#         cos_y, sin_y = self.get_cos_sin(D, max_pos[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
#         cos_x, sin_x = self.get_cos_sin(D, max_pos[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
#         # split features into three along the feature dimension, and apply rope1d on each half
#         t, y, x = tokens.chunk(3, dim=-1)
#         t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
#         y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
#         x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
#         tokens = torch.cat((t, y, x), dim=-1)
#         return tokens
