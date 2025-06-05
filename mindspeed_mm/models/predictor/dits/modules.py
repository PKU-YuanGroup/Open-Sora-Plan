import torch
from torch import nn
import torch_npu
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from typing import Any, Dict, Optional, Tuple

from megatron.legacy.model.rms_norm import RMSNorm
from megatron.legacy.model.layer_norm import LayerNorm
from diffusers.models.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding


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

        conditioning = (timesteps_emb + pooled_projections).float()
        if conditioning.dtype != torch.float32:
            raise ValueError("Conditioning embeddings must be float32.")

        return conditioning


class AdaNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        norm_cls: str = 'rms_norm',
    ):
        super().__init__()
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        config = core_transformer_config_from_args(args)
        config.sequence_parallel = False
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = tensor_parallel.SingleColumnParallelLinear(
            embedding_dim,
            output_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False
        )
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = LayerNorm
        self.norm = self.norm_cls(
            output_dim // 2, eps=norm_eps, sequence_parallel=self.sequence_parallel
        )

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)
        temb = self.linear(self.silu(temb))[0]
        if self.sequence_parallel:
            temb = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(temb)
        else:
            temb = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(temb)
        # x shape: (S B H), temb shape: (B, H)
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[None, :, :]
        scale = scale[None, :, :]
        weight_dtype = x.dtype
        with torch.autocast("cuda", enabled=False):
            x = self.norm(x).float() * (1 + scale.float()) + shift.float()
        return x.to(weight_dtype)

class OpenSoraNormZero(nn.Module):
    def __init__(
        self,
        timestep_embed_dim: int, 
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_cls: str = 'rms_norm', 
        context_pre_only: bool = False,
    ) -> None:
        super().__init__()
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        config = core_transformer_config_from_args(args)
        config.sequence_parallel = False
        
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = LayerNorm

        self.silu = nn.SiLU()
        self.linear = tensor_parallel.SingleColumnParallelLinear(
            timestep_embed_dim,
            6 * embedding_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False
        )
        self.norm = self.norm_cls(embedding_dim, eps=eps, sequence_parallel=self.sequence_parallel)
        self.norm_enc = None
        if not context_pre_only:
            self.norm_enc = self.norm_cls(embedding_dim, eps=eps, sequence_parallel=self.sequence_parallel)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temb = self.linear(self.silu(temb))[0]
        if self.sequence_parallel:
            temb = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(temb)
        else:
            temb = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(temb)
        shift, scale, gate, enc_shift, enc_scale, enc_gate = temb.chunk(6, dim=1)
        weight_dtype = hidden_states.dtype
        with torch.autocast("cuda", enabled=False):
            hidden_states = self.norm(hidden_states).float() * (1 + scale.float())[None, :, :] + shift.float()[None, :, :] # because hidden_states'shape is (S B H), so we need to add None at the first dimension
            if self.norm_enc is not None:
                encoder_hidden_states = self.norm_enc(encoder_hidden_states).float() * (1 + enc_scale.float())[None, :, :] + enc_shift.float()[None, :, :]
        return hidden_states.to(weight_dtype), encoder_hidden_states.to(weight_dtype), gate[None, :, :], enc_gate[None, :, :]
