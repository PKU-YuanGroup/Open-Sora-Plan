from typing import Tuple, Optional, List
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_npu
from einops import rearrange, repeat
from megatron import core
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from megatron.legacy.model.rms_norm import RMSNorm
from megatron.legacy.model.layer_norm import LayerNorm

from mindspeed_mm.utils.utils import video_to_image
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.embeddings.rope import RoPE3D, PositionGetter3D, apply_rotary_emb
from mindspeed_mm.models.common.conv import CausalConv3d, WfCausalConv3d

# TODO: 使用megatron通信接口替换
from .communications import (
    all_to_all,
    all_to_all_SBH,
    split_forward_gather_backward,
)

class MultiHeadSparseMMAttentionSBH(nn.Module):
    """
    A multi-head attention layer for both self-atten and cross-atten, layout "SBH".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        dropout: The dropout probability to use.
        proj_qkv_bias: Whether to use bias in qkv projection.
        proj_out_bias: Whether to use bias in out projection.
        interpolation_scale: The scale of interpolation.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        added_kv_proj_dim: int = None,
        dropout: float = 0.0,
        proj_qkv_bias: bool = False,
        proj_out_bias: bool = True,
        sparse1d: bool = False,
        sparse_n: int = None,
        sparse_group: bool = None,
        is_cross_attn: bool = False,
        context_pre_only:bool = None,
        qk_norm: Optional[str] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn

        self.context_pre_only = context_pre_only
        self.added_kv_proj_dim = added_kv_proj_dim
        

        args = get_args()
        config = core_transformer_config_from_args(args)
        self.sp_size = mpu.get_context_parallel_world_size()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()

        self.num_attention_heads_per_partition = core.utils.divide(num_heads, self.tp_size)
        self.num_attention_heads_per_partition_per_cp = core.utils.divide(self.num_attention_heads_per_partition,
                                                                          self.sp_size)
        
        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = LayerNorm(head_dim, eps=eps, sequence_parallel=True)
            self.norm_k = LayerNorm(head_dim, eps=eps, sequence_parallel=True)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
            self.norm_k = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm}")

        key_dim = key_dim if key_dim is not None else query_dim

        self.to_q = tensor_parallel.ColumnParallelLinear(
            query_dim,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_qkv_bias,
            gather_output=False
        )
        self.to_k = tensor_parallel.ColumnParallelLinear(
            key_dim,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_qkv_bias,
            gather_output=False
        )

        self.to_v = tensor_parallel.ColumnParallelLinear(
            key_dim,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_qkv_bias,
            gather_output=False
        )

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = tensor_parallel.ColumnParallelLinear(
                self.added_kv_proj_dim,
                self.inner_dim,
                config=config,
                init_method=config.init_method,
                bias=proj_qkv_bias,
                gather_output=False
            )
            self.add_v_proj = tensor_parallel.ColumnParallelLinear(
                self.added_kv_proj_dim,
                self.inner_dim,
                config=config,
                init_method=config.init_method,
                bias=proj_qkv_bias,
                gather_output=False
            )
            if self.context_pre_only is not None:
                self.add_q_proj = tensor_parallel.ColumnParallelLinear(
                    self.added_kv_proj_dim,
                    self.inner_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=proj_qkv_bias,
                    gather_output=False
                )

        self.to_out = nn.ModuleList(
            [
                tensor_parallel.RowParallelLinear(
                    query_dim,
                    self.inner_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=proj_out_bias,
                    input_is_parallel=True,
                    skip_bias_add=False
                ),
                nn.Dropout(dropout)
            ]
        )

        self.to_add_out = None
        if self.context_pre_only is not None:
            if self.tp_size > 1 or not self.context_pre_only:
                self.to_add_out = tensor_parallel.RowParallelLinear(
                    self.added_kv_proj_dim,
                    self.inner_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=proj_out_bias,
                    input_is_parallel=True,
                    skip_bias_add=False
                )

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "layer_norm":
                self.norm_added_q = LayerNorm(head_dim, eps=eps, sequence_parallel=True)
                self.norm_added_k = LayerNorm(head_dim, eps=eps, sequence_parallel=True)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
                self.norm_added_k = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
            else:
                raise ValueError(f"Unsupported qk_norm: {qk_norm}")

        

    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        _len = x.shape[0]
        if _len != frame * height * width:
            raise ValueError("shape mismatched.")
        pad_len = 0
        if _len % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - _len % (self.sparse_n * self.sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        if not self.sparse_group:
            x = rearrange(x, '(g k) b d -> g (k b) d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n m k) b d -> (n k) (m b) d', m=self.sparse_n, k=self.sparse_n)
        return x, pad_len

    def _reverse_sparse_1d(self, x, frame, height, width, pad_len):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        if x.shape[0] != (frame * height * width + pad_len) // self.sparse_n:
            raise ValueError("shape mismatched.")
        if not self.sparse_group:
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n k) (m b) d -> (n m k) b d', m=self.sparse_n, k=self.sparse_n)
        x = x[:frame * height * width, :, :]
        return x

    def _sparse_1d_enc(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.sparse_n)
        return x

    def _reverse_sparse_1d_enc(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = rearrange(x, 's (k b) d -> s k b d', k=self.sparse_n).mean(1)
        return x
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        frames: int, 
        height: int, 
        width: int, 
        attention_mask: Optional[torch.Tensor] = None,
        video_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: The hidden states of the visual stream.
            encoder_hidden_states: The hidden states of the textual stream.
            frames: The frame number of video
            height: The height of the video
            width: The width of the video
            attention_mask: The attention mask to use.
            video_rotary_emb: The rotary embeddings for the video
        """
        # Step 1: Project the hidden states and encoder hidden states
        q = self.to_q(hidden_states)[0]
        k = self.to_k(hidden_states)[0]
        v = self.to_v(hidden_states)[0]
        added_q = self.add_q_proj(encoder_hidden_states)[0]
        added_k = self.add_k_proj(encoder_hidden_states)[0]
        added_v = self.add_v_proj(encoder_hidden_states)[0]
        visual_sequence_length, batch_size, _ = q.shape
        text_sequence_length_length, batch_size, _ = added_q.shape

        total_frames = frames

        if self.sp_size > 1:
            q = q.view(-1, self.num_attention_heads_per_partition, self.head_dim)
            k = k.view(-1, self.num_attention_heads_per_partition, self.head_dim)
            v = v.view(-1, self.num_attention_heads_per_partition, self.head_dim)
            total_frames = frames * self.sp_size
            sp_group = mpu.get_context_parallel_group()

            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            q = all_to_all_SBH(q, sp_group, scatter_dim=1, gather_dim=0)
            k = all_to_all_SBH(k, sp_group, scatter_dim=1, gather_dim=0)
            v = all_to_all_SBH(v, sp_group, scatter_dim=1, gather_dim=0)

            added_q = added_q.view(-1, self.num_attention_heads_per_partition, self.head_dim)
            added_k = added_k.view(-1, self.num_attention_heads_per_partition, self.head_dim)
            added_v = added_v.view(-1, self.num_attention_heads_per_partition, self.head_dim)

            added_q = all_to_all_SBH(added_q, sp_group, scatter_dim=1, gather_dim=0)
            added_k = all_to_all_SBH(added_k, sp_group, scatter_dim=1, gather_dim=0)
            added_v = all_to_all_SBH(added_v, sp_group, scatter_dim=1, gather_dim=0)

        # Step 2: QK Norm
        q = q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        k = k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)

        added_q = added_q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        added_k = added_k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        added_q = self.norm_added_q(added_q)
        added_k = self.norm_added_k(added_k)

        # Step 3: Apply rope
        q = apply_rotary_emb(q, video_rotary_emb)
        k = apply_rotary_emb(k, video_rotary_emb)

        q = q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        k = k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        v = v.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)

        added_q = added_q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        added_k = added_k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        added_v = added_v.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)

        # Step 4: Sparse 1D
        if self.sparse1d:
            q, pad_len = self._sparse_1d(q, total_frames, height, width)
            k, _ = self._sparse_1d(k, total_frames, height, width)
            v, _ = self._sparse_1d(v, total_frames, height, width)
            added_q = self._sparse_1d_enc(added_q)
            added_k = self._sparse_1d_enc(added_k)
            added_v = self._sparse_1d_enc(added_v)

        # Step 5: Concat hidden_states and encoder_hidden_states to do mm attention
        fa_visual_sequence_length = q.shape[0]
        fa_text_sequence_length_length = added_q.shape[0]
        q = torch.cat([q, added_q], dim=0)
        k = torch.cat([k, added_k], dim=0)
        v = torch.cat([v, added_v], dim=0)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_attention_heads_per_partition_per_cp,
            atten_mask=attention_mask,
            input_layout="SBH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        hidden_states, encoder_hidden_states = out.split([fa_visual_sequence_length, fa_text_sequence_length_length], dim=0)

        # Step 6: Reverse sparse 1D
        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frames, height, width, pad_len)
            encoder_hidden_states = self._reverse_sparse_1d_enc(encoder_hidden_states)

        # [s, b, h // sp * d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
        if self.sp_size > 1:
            sp_group = mpu.get_context_parallel_group()
            hidden_states = all_to_all_SBH(
                hidden_states.reshape(-1, self.num_attention_heads_per_partition_per_cp, self.head_dim),
                sp_group, scatter_dim=0, gather_dim=1)
            hidden_states = hidden_states.view(visual_sequence_length, batch_size, -1)

            encoder_hidden_states = all_to_all_SBH(
                encoder_hidden_states.reshape(-1, self.num_attention_heads_per_partition_per_cp, self.head_dim),
                sp_group, scatter_dim=0, gather_dim=1)
            encoder_hidden_states = encoder_hidden_states.view(text_sequence_length_length, batch_size, -1)
        
        # Step 7: Project out
        hidden_states = self.to_out[0](hidden_states)[0]
        hidden_states = self.to_out[1](hidden_states)

        if self.to_add_out is not None:
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)[0]

        return hidden_states, encoder_hidden_states


class MultiHeadAttentionBSH(nn.Module):
    """
    A multi-head attention layer for both self-atten and cross-atten, layout "BSH".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        dropout: The dropout probability to use.
        proj_qkv_bias: Whether to use bias in qkv projection.
        proj_out_bias: Whether to use bias in out projection.
        use_rope: Whether to use rope
        interpolation_scale: The scale of interpolation.

    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        proj_qkv_bias: bool = False,
        proj_out_bias: bool = True,
        use_rope: bool = False,
        interpolation_scale: Tuple[int] = (1, 1, 1),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE3D(interpolation_scale=interpolation_scale)
            self.position_getter = PositionGetter3D(atten_layout="BSH")

        key_dim = key_dim if key_dim is not None else query_dim

        self.proj_q = nn.Linear(query_dim, self.inner_dim, bias=proj_qkv_bias)
        self.proj_k = nn.Linear(key_dim, self.inner_dim, bias=proj_qkv_bias)
        self.proj_v = nn.Linear(key_dim, self.inner_dim, bias=proj_qkv_bias)

        self.proj_out = nn.Linear(self.inner_dim, query_dim, bias=proj_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: The hidden states of the query.
            key: The hidden states of the key.
            mask: The attention mask to use.
            **kwargs: Additional keyword arguments to pass along
        """
        input_ndim = query.ndim
        if input_ndim == 4:
            b, c, h, w = query.shape
            query = query.view(b, c, h * w).transpose(1, 2)

        key = query if key is None else key
        b, _, _ = query.shape

        if mask is not None:
            mask = mask.view(b, 1, -1, mask.shape[-1])

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(key)

        q = q.view(b, -1, self.num_heads, self.head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim)

        if self.use_rope:
            if (frames is None) or (height is None) or (width is None):
                raise ValueError(
                    "frames, height, and width can not be none when use_rope"
                )
            pos_thw = self.position_getter(
                b, t=frames, h=height, w=width, device=query.device
            )
            q = self.rope(q, pos_thw)
            k = self.rope(k, pos_thw)
        q = q.view(b, -1, self.inner_dim)
        k = k.view(b, -1, self.inner_dim)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_heads,
            atten_mask=mask,
            input_layout="BSH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        out = self.proj_out(out)
        out = self.dropout(out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        return out


class ParallelMultiHeadAttentionSBH(nn.Module):
    """
    A multi-head context parallel attention layer for both self-attention and cross-attention, layout "SBH".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        dropout: The dropout probability to use.
        proj_qkv_bias: Whether to use bias in qkv projection.
        proj_out_bias: Whether to use bias in out projection.
        use_rope: Whether to use rope
        interpolation_scale: The scale of interpolation.

    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        proj_qkv_bias: bool = False,
        proj_out_bias: bool = True,
        use_rope: bool = False,
        interpolation_scale: Tuple[int] = (1, 1, 1)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE3D(interpolation_scale=interpolation_scale)
            self.position_getter = PositionGetter3D(atten_layout="SBH")

        key_dim = key_dim if key_dim is not None else query_dim

        self.proj_q = nn.Linear(query_dim, self.inner_dim, bias=proj_qkv_bias)
        self.proj_k = nn.Linear(key_dim, self.inner_dim, bias=proj_qkv_bias)
        self.proj_v = nn.Linear(key_dim, self.inner_dim, bias=proj_qkv_bias)

        self.proj_out = nn.Linear(self.inner_dim, query_dim, bias=proj_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            query: The hidden states of the query.
            key: The hidden states of the key.
            mask: The attention mask to use.
            frames: The frame number of latents
            height: The height of the frame
            width: The width of the frame
        """
        if len(query.shape) != 3:
            raise AssertionError("Parallel attention only support SBH.")

        is_cross_attention = key is not None

        key = query if key is None else key
        s, b, _ = query.shape

        if mask is not None:
            mask = mask.view(b, 1, -1, mask.shape[-1])

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(key)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        sp_size = mpu.get_context_parallel_world_size()
        h_size_sp = self.inner_dim // sp_size
        sp_group = mpu.get_context_parallel_group()

        q = all_to_all_SBH(q, sp_group, scatter_dim=1, gather_dim=0).view(-1, b, h_size_sp)
        if not is_cross_attention:
            k = all_to_all_SBH(k, sp_group, scatter_dim=1, gather_dim=0).view(-1, b, h_size_sp)
            v = all_to_all_SBH(v, sp_group, scatter_dim=1, gather_dim=0).view(-1, b, h_size_sp)
        else:
            k = split_forward_gather_backward(k, sp_group, dim=1, grad_scale="down").view(-1, b, h_size_sp)
            v = split_forward_gather_backward(v, sp_group, dim=1, grad_scale="down").view(-1, b, h_size_sp)

        if self.use_rope:
            # TODO: 原仓BUG，view使用错误，不能跨轴view
            q = q.view(-1, b, self.num_heads // sp_size, self.head_dim)
            k = k.view(-1, b, self.num_heads // sp_size, self.head_dim)

            if (frames is None) or (height is None) or (width is None):
                raise ValueError("frames, height and width can not be none when use_rope")
            pos_thw = self.position_getter(b, t=frames, h=height, w=width, device=query.device)
            q = self.rope(q, pos_thw)
            k = self.rope(k, pos_thw)

        q = q.view(-1, b, h_size_sp)
        k = k.view(-1, b, h_size_sp)
        v = v.view(-1, b, h_size_sp)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_heads // sp_size,
            atten_mask=mask,
            input_layout="SBH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        out = out.view(-1, self.num_heads // sp_size, self.head_dim)
        out = all_to_all_SBH(out, sp_group, scatter_dim=0, gather_dim=1).view(-1, b, self.inner_dim)

        out = self.proj_out(out)
        out = self.dropout(out)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise AssertionError(
                "dim (%d) must be divisible by num_heads (%d)" % (dim, num_heads)
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def npu_spatial_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        B, N, _ = qkv.shape
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch_npu.npu_fusion_attention(
            q, k, v, self.num_heads, input_layout="BSND",
            pse=None,
            scale=self.scale,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1. - self.attn_drop.p if self.training else 1.,
            sync=False,
            inner_precise=0
        )[0]

        x = x.view(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def npu_temporal_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        B, N, _ = qkv.shape
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm_legacy:
            q, k = self.rotary_emb(q), self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            q, k = self.rotary_emb(q), self.rotary_emb(k)

        x = torch_npu.npu_fusion_attention(
            q, k, v, self.num_heads, input_layout="BNSD",
            pse=None,
            scale=self.scale,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1. - self.attn_drop.p if self.training else 1.,
            sync=False,
            inner_precise=0,
        )[0]

        x = x.transpose(1, 2)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flashattn = self.enable_flashattn
        qkv = self.qkv(x)

        if enable_flashattn:
            if qkv.dtype in [torch.float16, torch.bfloat16]:
                if self.rope:
                    return self.npu_temporal_attention(qkv)
                else:
                    return self.npu_spatial_attention(qkv)
            else:
                raise ValueError("The dtype of x must be torch.float16 or torch.bfloat16, got torch.float32 instead.")

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        attn = self.attn_drop(attn)
        x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flashattn=enable_flashattn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = (
            x.shape
        )  # for sequence parallel here, the N is a local sequence length
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape)

        # sp_group = get_sequence_parallel_group()
        sp_group = mpu.get_context_parallel_group()

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flashattn:
            # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.permute(qkv_permute_shape)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flashattn and q.dtype in [torch.float16, torch.bfloat16]:
            x = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                q.shape[-2],
                input_layout="BSND",
                pse=None,
                scale=self.scale,
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0 - self.attn_drop.p if self.training else 1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flashattn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise AssertionError(
                "d_model (%d) must be divisible by num_heads (%d)"
                % (d_model, num_heads)
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        if x.dtype not in [torch.float16, torch.bfloat16]:
            raise AssertionError("QKV's dtype must be in bf16 or fp16")
        q = self.q_linear(x).view(-1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        actual_seq_qlen = []
        actual_seq_kvlen = []
        if mask is not None:
            ans = 0
            for _ in range(B):
                ans += N
                actual_seq_qlen.append(ans)
            ans = 0
            for m in mask:
                ans += m
                actual_seq_kvlen.append(ans)
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            self.num_heads,
            input_layout="TND",
            pse=None,
            scale=1.0 / math.sqrt(self.head_dim),
            pre_tockens=65536,
            next_tockens=65536,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(actual_seq_kvlen),
            keep_prob=1.0 - self.attn_drop.p,
            sparse_mode=0,
        )[0]

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = mpu.get_context_parallel_group()
        sp_size = mpu.get_context_parallel_world_size()
        B, SUB_N, C = x.shape
        N = SUB_N * sp_size

        # shape:
        # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

        k = split_forward_gather_backward(
            k, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
        )
        v = split_forward_gather_backward(
            v, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
        )

        if x.dtype not in [torch.float16, torch.bfloat16]:
            raise AssertionError("QKV's dtype must be in bf16 or fp16")
        q = q.view(-1, self.num_heads // sp_size, self.head_dim)
        k = k.view(-1, self.num_heads // sp_size, self.head_dim)
        v = v.view(-1, self.num_heads // sp_size, self.head_dim)

        actual_seq_qlen = []
        actual_seq_kvlen = []
        if mask is not None:
            ans = 0
            for _ in range(B):
                ans += N
                actual_seq_qlen.append(ans)
            ans = 0
            for m in mask:
                ans += m
                actual_seq_kvlen.append(ans)
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[-2],
            input_layout="TND",
            pse=None,
            scale=1.0 / math.sqrt(self.head_dim),
            pre_tockens=65536,
            next_tockens=65536,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(actual_seq_kvlen),
            keep_prob=1.0 - self.attn_drop.p,
            sparse_mode=0,
        )[0]

        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Conv2dAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=32,
        eps=1e-6,
        kernel_size=1,
        stride=1,
        padding=0,
        affine=True,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine
        )
        self.q = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.k = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.v = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

    @video_to_image
    def forward(self, x):
        y = x
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # [b, hw, c]
        k = k.reshape(b, c, h * w)  # [b, c, hw]
        z = torch.bmm(q, k)  # [b, hw, hw]
        z = z * (int(c) ** (-0.5))
        z = torch.nn.functional.softmax(z, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        z = z.permute(0, 2, 1)  # [b, hw, hw] (first hw of k, second of q)
        y = torch.bmm(v, z)  # [b, c, hw] (hw of q)
        y = y.reshape(b, c, h, w)

        y = self.proj_out(y)

        return x + y


class CausalConv3dAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine
        )
        self.q = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.k = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.v = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.proj_out = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        y = x
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # compute attention
        # q: (b c t h w) -> (b t c h w) -> (b*t c h*w) -> (b*t h*w c)
        b, c, t, h, w = q.shape
        q = torch_npu.npu_confusion_transpose(
            q, (0, 2, 1, 3, 4), (b * t, c, h * w), True
        )
        q = q.permute(0, 2, 1)

        # k: (b c t h w) -> (b t c h w) -> (b*t c h*w)
        k = torch_npu.npu_confusion_transpose(
            k, (0, 2, 1, 3, 4), (b * t, c, h * w), True
        )

        # w: (b*t hw hw)
        z = torch.bmm(q, k)
        z = z * (int(c) ** (-0.5))
        z = torch.nn.functional.softmax(z, dim=2)

        # attend to values
        # v: (b c t h w) -> (b t c h w) -> (bt c hw)
        # z: (bt hw hw) -> (bt hw hw)
        v = torch_npu.npu_confusion_transpose(v, (0, 2, 1, 3, 4), (b * t, c, h * w), True)
        z = z.permute(0, 2, 1)  # [b, hw, hw] (first hw of k, second of q)
        y = torch.bmm(v, z)  # [b, c, hw] (hw of q)

        # y: (b*t c hw) -> (b t c h w) -> (b c t h w)
        y = torch_npu.npu_confusion_transpose(y, (0, 2, 1, 3, 4), (b, t, c, h, w), False)

        y = self.proj_out(y)

        return x + y


class WfCausalConv3dAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.norm = normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.q = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.k = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.v = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.proj_out = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()

        attn_output = self.run_attention(q, k, v, atten_mask=None, input_layout="BSH", head_dim=c, head_num=1, enable_FA=c<=512)

        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_
    
    def run_attention(self, query, key, value, atten_mask, input_layout, head_dim, head_num, enable_FA=True):
        if enable_FA:
            hidden_states = torch_npu.npu_fusion_attention(query, key, value,
                                                           atten_mask=atten_mask,
                                                           input_layout=input_layout,
                                                           scale=1 / math.sqrt(head_dim),
                                                           head_num=head_num)[0]
        else:
            hidden_states = self.scaled_dot_product_attention(query, key, value,
                                                              atten_mask=atten_mask,
                                                              input_layout=input_layout,
                                                              scale=1 / math.sqrt(head_dim),
                                                              head_num=head_num)
        return hidden_states

    def scaled_dot_product_attention(self, query, key, value, input_layout, head_num=None,
                                     atten_mask=None, scale=None, dropout_p=0.0, is_causal=False) -> torch.Tensor:
        def trans_tensor_shape(x, layout, head_num):
            if layout == "BSH":
                batch = x.shape[0]
                x = x.view(batch, -1, head_num, x.shape[-1] // head_num).transpose(1, 2).contiguous()
            elif layout == "SBH":
                batch = x.shape[1]
                x = x.view(-1, batch * head_num, x.shape[-1] // head_num).transpose(0, 1).contiguous()
                x = x.view(batch, head_num, -1, x.shape[-1])
            return x

        query = trans_tensor_shape(query, input_layout, head_num)
        key = trans_tensor_shape(key, input_layout, head_num)
        value = trans_tensor_shape(value, input_layout, head_num)

        attn_weight = query @ key.transpose(-2, -1) * scale
        attn_bias = torch.zeros_like(attn_weight, dtype=query.dtype, device=query.device)
        if is_causal:
            assert atten_mask is None
            temp_mask = torch.zeros_like(attn_weight, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), -10000.0)
            attn_bias.to(query.dtype)

        if atten_mask is not None:
            assert (not self.enable_FA) and atten_mask.dtype != torch.bool, \
                "attention_mask must not be bool type when use this function"

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value
        if input_layout == "BSH":
            output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, head_num * output.shape[-1])
        else:
            output = output.view(output.shape[0] * head_num, -1, output.shape[-1]).transpose(0, 1).contiguous()
            output = output.view(output.shape[0], -1, head_num * output.shape[-1])
        return output

class WhisperAttention(nn.Module):
    """
    A multi-head attention layer for both self-atten and cross-atten, layout "BSH".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        dropout: The dropout probability to use.
        proj_qkv_bias: Whether to use bias in qkv projection.
        proj_out_bias: Whether to use bias in out projection.
        use_rope: Whether to use rope
        interpolation_scale: The scale of interpolation.

    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        proj_qv_bias: bool = False,
        proj_out_bias: bool = True,
        interpolation_scale: Tuple[int] = (1, 1, 1),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim

        key_dim = key_dim if key_dim is not None else query_dim

        self.proj_q = nn.Linear(query_dim, self.inner_dim, bias=proj_qv_bias)
        self.proj_k = nn.Linear(key_dim, self.inner_dim, bias=False)
        self.proj_v = nn.Linear(key_dim, self.inner_dim, bias=proj_qv_bias)

        self.proj_out = nn.Linear(self.inner_dim, query_dim, bias=proj_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: The hidden states of the query.
            key: The hidden states of the key.
            mask: The attention mask to use.
            **kwargs: Additional keyword arguments to pass along
        """
        input_ndim = query.ndim
        if input_ndim == 4:
            b, c, h, w = query.shape
            query = query.view(b, c, h * w).transpose(1, 2)

        key = query if key is None else key
        b, _, _ = query.shape

        if mask is not None:
            mask = mask.view(b, 1, -1, mask.shape[-1])

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(key)

        q = q.view(b, -1, self.inner_dim)
        k = k.view(b, -1, self.inner_dim)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_heads,
            atten_mask=mask,
            input_layout="BSH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        out = self.proj_out(out)
        out = self.dropout(out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        return out
