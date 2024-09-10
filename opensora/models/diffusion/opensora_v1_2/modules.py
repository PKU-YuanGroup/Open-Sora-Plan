
import torch
from torch import nn
from typing import Optional, Tuple
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward

from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from ..common import Attention

logger = logging.get_logger(__name__)


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
            motion_score = motion_score.repeat(batch_size//motion_score.shape[0])
            assert motion_score.shape[0] == batch_size
        # No modulation happening here.
        embedded_motion = self.emb(motion_score, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_motion))
    

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
        sparse1d: bool = False,
        sparse_n: int = 2,
        sparse_group: bool = False,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            interpolation_scale_thw=interpolation_scale_thw, 
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=False,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            interpolation_scale_thw=interpolation_scale_thw, 
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=True,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Scale-shift.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> torch.FloatTensor:
        
        # 0. Self-Attention
        batch_size = hidden_states.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
        ).chunk(6, dim=0)

        norm_hidden_states = self.norm1(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask, frame=frame, height=height, width=width, 
        )

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        norm_hidden_states = hidden_states

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask, frame=frame, height=height, width=width,
        )
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states
