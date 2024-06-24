import torch
import torch.nn as nn
import torch.nn.functional as F

from opensora.models.diffusion.utils.pos_embed import RoPE1D, RoPE2D, LinearScalingRoPE2D, LinearScalingRoPE1D

from opensora.models.diffusion.latte.modules import Attention

from typing import Any, Dict, Optional, Tuple, Callable
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, is_xformers_available

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# IP Adapter
class VideoIPAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        dim=1152, 
        attention_mode='xformers', 
        use_rope=False, 
        rope_scaling=None, 
        compress_kv_factor=None,
        
        num_vip_tokens=128,
        vip_scale=1.0,
        dropout=0.0,
    ):
        self.dim = dim
        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.compress_kv_factor = compress_kv_factor

        if self.use_rope:
            self._init_rope()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.to_k_vip = nn.Linear(dim, dim, bias=False)
        self.to_v_vip = nn.Linear(dim, dim, bias=False)

        self.to_out_vip = nn.ModuleList(
            [
                zero_module(nn.Linear(dim, dim, bias=False)),
                nn.Dropout(dropout),
            ]
        )
        
        self.num_vip_tokens = num_vip_tokens
        self.vip_scale = vip_scale

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rope2d = RoPE2D()
            self.rope1d = RoPE1D()
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor_2d = self.rope_scaling["factor_2d"]
            scaling_factor_1d = self.rope_scaling["factor_1d"]
            if scaling_type == "linear":
                self.rope2d = LinearScalingRoPE2D(scaling_factor=scaling_factor_2d)
                self.rope1d = LinearScalingRoPE1D(scaling_factor=scaling_factor_1d)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
            
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        position_q: Optional[torch.LongTensor] = None,
        position_k: Optional[torch.LongTensor] = None,
        last_shape: Tuple[int] = None, 
    ) -> torch.FloatTensor:
        
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)


        if self.compress_kv_factor is not None:
            batch_size = hidden_states.shape[0]
            if len(last_shape) == 2:
                encoder_hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, self.dim, *last_shape)
                encoder_hidden_states = attn.sr(encoder_hidden_states).reshape(batch_size, self.dim, -1).permute(0, 2, 1)
            elif len(last_shape) == 1:
                encoder_hidden_states = hidden_states.permute(0, 2, 1)
                if last_shape[0] % 2 == 1:
                    first_frame_pad = encoder_hidden_states[:, :, :1].repeat((1, 1, attn.kernel_size - 1))
                    encoder_hidden_states = torch.concatenate((first_frame_pad, encoder_hidden_states), dim=2)
                encoder_hidden_states = attn.sr(encoder_hidden_states).permute(0, 2, 1)
            else:
                raise NotImplementedError(f'NotImplementedError with last_shape {last_shape}')
                
            encoder_hidden_states = attn.norm(encoder_hidden_states)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # import ipdb;ipdb.set_trace()
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_vip_tokens
            encoder_hidden_states, vip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        vip_key = self.to_k_vip(vip_hidden_states)
        vip_value = self.to_v_vip(vip_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads


        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        vip_key = vip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        vip_value = vip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.use_rope:
            # require the shape of (batch_size x nheads x ntokens x dim)
            if position_q.ndim == 3:
                query = self.rope2d(query, position_q)
            elif position_q.ndim == 2:
                query = self.rope1d(query, position_q)
            else:
                raise NotImplementedError
            if position_k.ndim == 3:
                key = self.rope2d(key, position_k)
                vip_key = self.rope2d(vip_key, position_k)
            elif position_k.ndim == 2:
                key = self.rope1d(key, position_k)
                vip_key = self.rope1d(vip_key, position_k)
            else:
                raise NotImplementedError

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.attention_mode == 'flash':
            assert attention_mask is None or torch.all(attention_mask.bool()), 'flash-attn do not support attention_mask'
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                vip_hidden_states = F.scaled_dot_product_attention(
                    query, vip_key, vip_value, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'xformers':
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                vip_hidden_states = F.scaled_dot_product_attention(
                    query, vip_key, vip_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'math':
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            vip_hidden_states = F.scaled_dot_product_attention(
                query, vip_key, vip_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        vip_hidden_states = vip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        vip_hidden_states = vip_hidden_states.to(query.dtype)

        # linear proj
        vip_hidden_states = self.to_out_vip[0](vip_hidden_states)
        # dropout
        vip_hidden_states = self.to_out_vip[1](vip_hidden_states)

        hidden_states = hidden_states + self.vip_scale * vip_hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

