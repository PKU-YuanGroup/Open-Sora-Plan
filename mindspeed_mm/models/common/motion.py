from diffusers.models.embeddings import SinusoidalPositionalEmbedding, PixArtAlphaTextProjection, Timesteps, TimestepEmbedding
from torch import nn
import torch
from typing import Optional, Tuple

class MotionEmbeddings(nn.Module):
    """
    From PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.motion_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.motion_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, motion_score, hidden_dtype):
        motions_proj = self.motion_proj(motion_score)
        motions_emb = self.motion_embedder(motions_proj.to(dtype=hidden_dtype))  # (N, D)
        return motions_emb

class MotionAdaLayerNormSingle(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.emb = MotionEmbeddings(embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

        self.linear.weight.data.zero_()
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(
            self,
            motion_score: torch.Tensor,
            batch_size: int,
            hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(motion_score, float) or isinstance(motion_score, int):
            motion_score = torch.tensor([motion_score], device=self.linear.weight.device)
        if motion_score.ndim == 2:
            assert motion_score.shape[1] == 1
            motion_score = motion_score.squeeze(1)
        assert motion_score.ndim == 1
        if motion_score.shape[0] != batch_size:
            motion_score = motion_score.repeat(batch_size // motion_score.shape[0])
            assert motion_score.shape[0] == batch_size
        # No modulation happening here.
        embedded_motion = self.emb(motion_score, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_motion))