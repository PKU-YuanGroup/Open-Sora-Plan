
import torch
from torch import nn
from typing import Optional, Tuple
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward

from diffusers.models.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding

from ..common import Attention

logger = logging.get_logger(__name__)

class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, timestep_embed_dim, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=timestep_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=timestep_embed_dim, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning

class OpenSoraLayerNormZero(nn.Module):
    def __init__(
        self,
        timestep_embed_dim: int, 
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_enc = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[None, :, :] + shift[None, :, :]
        encoder_hidden_states = self.norm_enc(encoder_hidden_states) * (1 + enc_scale)[None, :, :] + enc_shift[None, :, :]
        return hidden_states, encoder_hidden_states, gate[None, :, :], enc_gate[None, :, :]

class OpenSoraLayerNormZeroMlp(nn.Module):
    def __init__(
        self,
        timestep_embed_dim: int, 
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[None, :, :] + shift[None, :, :]
        return hidden_states, gate[None, :, :]
    
@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        timestep_embed_dim: int, 
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
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group

        # 1. Self-Attn
        self.norm1 = OpenSoraLayerNormZero(timestep_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim, 
            qk_norm="layer_norm",
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
            qk_norm="layer_norm",
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
        self.norm3 = OpenSoraLayerNormZeroMlp(timestep_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> torch.FloatTensor:
        
        # 0. Self-Attention
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, gate_mca = self.norm1(
            hidden_states, encoder_hidden_states, embedded_timestep
            )
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask, frame=frame, height=height, width=width, 
        )
        attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states

        # 1. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=encoder_attention_mask, frame=frame, height=height, width=width,
        )
        attn_output = gate_mca * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Feed-forward
        norm_hidden_states, gate_mlp = self.norm3(hidden_states, embedded_timestep)
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output
        hidden_states = ff_output + hidden_states

        return hidden_states
