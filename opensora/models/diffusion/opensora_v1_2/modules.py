import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward

from diffusers.models.attention_processor import Attention as Attention_
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
try:
    import torch_npu
    from opensora.npu_config import npu_config, set_run_dtype
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info as xccl_info
    from opensora.acceleration.communications import all_to_all_SBH
except:
    torch_npu = None
    npu_config = None
    set_run_dtype = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info as xccl_info
    from opensora.utils.communications import all_to_all_SBH

from ..common import RoPE3D, PositionGetter3D

logger = logging.get_logger(__name__)


class Attention(Attention_):
    def __init__(
            self, interpolation_scale_thw, sparse1d, sparse_n, 
            sparse_group, is_cross_attn, **kwags
            ):
        processor = OpenSoraAttnProcessor2_0(
            interpolation_scale_thw=interpolation_scale_thw, sparse1d=sparse1d, sparse_n=sparse_n, 
            sparse_group=sparse_group, is_cross_attn=is_cross_attn
            )
        super().__init__(processor=processor, **kwags)

    @staticmethod
    def prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n, head_num):
        attention_mask = attention_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        l = attention_mask.shape[-1]
        if l % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - l % (sparse_n * sparse_n)

        attention_mask_sparse = F.pad(attention_mask, (0, pad_len, 0, 0), value=-9980.0)
        attention_mask_sparse_1d = rearrange(
            attention_mask_sparse, 
            'b 1 1 (g k) -> (k b) 1 1 g', 
            k=sparse_n
            )
        attention_mask_sparse_1d_group = rearrange(
            attention_mask_sparse, 
            'b 1 1 (n m k) -> (m b) 1 1 (n k)',
            m=sparse_n, 
            k=sparse_n
            )
        encoder_attention_mask_sparse = encoder_attention_mask.repeat(sparse_n, 1, 1, 1)
        if npu_config is not None:
            attention_mask_sparse_1d = npu_config.get_attention_mask(
                attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1]
                )
            attention_mask_sparse_1d_group = npu_config.get_attention_mask(
                attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1]
                )
            
            encoder_attention_mask_sparse_1d = npu_config.get_attention_mask(
                encoder_attention_mask_sparse, attention_mask_sparse_1d.shape[-1]
                )
            encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d
        else:
            attention_mask_sparse_1d = attention_mask_sparse_1d.repeat_interleave(head_num, dim=1)
            attention_mask_sparse_1d_group = attention_mask_sparse_1d_group.repeat_interleave(head_num, dim=1)

            encoder_attention_mask_sparse_1d = encoder_attention_mask_sparse.repeat_interleave(head_num, dim=1)
            encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d

        return {
                    False: (attention_mask_sparse_1d, encoder_attention_mask_sparse_1d),
                    True: (attention_mask_sparse_1d_group, encoder_attention_mask_sparse_1d_group)
                }

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if get_sequence_parallel_state():
            head_size = head_size // xccl_info.world_size  # e.g, 24 // 8
        
        if attention_mask is None:  # b 1 t*h*w in sa, b 1 l in ca
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            print(f'attention_mask.shape, {attention_mask.shape}, current_length, {current_length}, target_length, {target_length}')
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

class OpenSoraAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, interpolation_scale_thw=(1, 1, 1), 
                 sparse1d=False, sparse_n=2, sparse_group=False, is_cross_attn=True):
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        
        self._init_rope(interpolation_scale_thw)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        l = x.shape[0]
        assert l == frame*height*width
        pad_len = 0
        if l % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - l % (self.sparse_n * self.sparse_n)
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
        assert x.shape[0] == (frame*height*width+pad_len) // self.sparse_n
        if not self.sparse_group:
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n k) (m b) d -> (n m k) b d', m=self.sparse_n, k=self.sparse_n)
        x = x[:frame*height*width, :, :]
        return x
    
    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.sparse_n)
        return x
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states

        sequence_length, batch_size, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # if attention_mask is not None:
        #     if npu_config is None:
        #         # scaled_dot_product_attention expects attention_mask shape to be
        #         # (batch, heads, source_length, target_length)
        #         if get_sequence_parallel_state():
        #             # sequence_length has been split, so we need sequence_length * nccl_info.world_size
        #             # (sp*b 1 s), where s has not been split
        #             # (sp*b 1 s) -prepare-> (sp*b*head 1 s) -> (sp*b head 1 s), where head has been split (e.g, 24 // 8)
        #             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length * xccl_info.world_size, batch_size)
        #             attention_mask = attention_mask.view(batch_size, attn.heads // xccl_info.world_size, -1, attention_mask.shape[-1])
        #         else:
        #             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        FA_head_num = attn.heads
        total_frame = frame

        if get_sequence_parallel_state():
            sp_size = xccl_info.world_size
            FA_head_num = attn.heads // sp_size
            total_frame = frame * sp_size
            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            query = all_to_all_SBH(query.view(-1, attn.heads, head_dim), scatter_dim=1, gather_dim=0)
            key = all_to_all_SBH(key.view(-1, attn.heads, head_dim), scatter_dim=1, gather_dim=0)
            value = all_to_all_SBH(value.view(-1, attn.heads, head_dim), scatter_dim=1, gather_dim=0)
        query = query.view(-1, batch_size, FA_head_num, head_dim)
        key = key.view(-1, batch_size, FA_head_num, head_dim)

        if not self.is_cross_attn:
            # require the shape of (ntokens x batch_size x nheads x dim)
            pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width, device=query.device)

            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)
            
            # query = rearrange(query, 's b h d -> b h s d')
            # key = rearrange(key, 's b h d -> b h s d')
            # dtype = query.dtype

            # query = self.rope(query.to(torch.float16), pos_thw)
            # key = self.rope(key.to(torch.float16), pos_thw)

            # query = rearrange(query, 'b h s d -> s b h d').to(dtype)
            # key = rearrange(key, 'b h s d -> s b h d').to(dtype)

        query = query.view(-1, batch_size, FA_head_num * head_dim)
        key = key.view(-1, batch_size, FA_head_num * head_dim)
        value = value.view(-1, batch_size, FA_head_num * head_dim)
        # print(f'q {query.shape}, k {key.shape}, v {value.shape}')
        if self.sparse1d:
            query, pad_len = self._sparse_1d(query, total_frame, height, width)
            if self.is_cross_attn:
                key = self._sparse_1d_kv(key)
                value = self._sparse_1d_kv(value)
            else:
                key, pad_len = self._sparse_1d(key, total_frame, height, width)
                value, pad_len = self._sparse_1d(value, total_frame, height, width)

        # print(f'after sparse q {query.shape}, k {key.shape}, v {value.shape}')
        if npu_config is not None:
            hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH", head_dim, FA_head_num)
        else:
            query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
            key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
            value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
            # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
            # 0, 0 ->(bool) False, False ->(any) False ->(not) True
            # if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
            #     attention_mask = None
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=FA_head_num)

        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frame, height, width, pad_len)

        # [s, b, h // sp * d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
        if get_sequence_parallel_state():
            hidden_states = all_to_all_SBH(hidden_states.reshape(-1, FA_head_num, head_dim), scatter_dim=0, gather_dim=1)
            hidden_states = hidden_states.view(-1, batch_size, inner_dim)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if attn.residual_connection:
        #     print('attn.residual_connection')
            # hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


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
