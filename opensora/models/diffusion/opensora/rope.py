import torch

class PositionGetter3D(object):
    """ return positions of patches """

    def __init__(self, interpolation_scale_thw):
        self.interpolation_scale_thw = interpolation_scale_thw
        self.cache_positions = {}
        
    def __call__(self, b, t, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device) / self.interpolation_scale_thw[2]
            y = torch.arange(h, device=device) / self.interpolation_scale_thw[1]
            t = torch.arange(t, device=device) / self.interpolation_scale_thw[0]
            self.cache_positions[t,h,w] = torch.cartesian_prod(t, y, x) # (t, h, w, 3)
        pos = self.cache_positions[t,h,w].view(1, t*h*w, 3).expand(b, -1, 3).clone()
        return pos
    

class RoPE3D(torch.nn.Module):

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
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        assert positions.ndim == 3 and positions.shape[-1] == 3  # Batch, Seq, 3
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, positions[:, :, 0], cos, sin)
        y = self.apply_rope1d(y, positions[:, :, 1], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 2], cos, sin)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens
