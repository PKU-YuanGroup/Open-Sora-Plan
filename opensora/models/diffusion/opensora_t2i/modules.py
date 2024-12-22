
import torch
from torch import nn
from typing import Optional, Tuple
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from torch.nn import functional as F
from diffusers.models.normalization import RMSNorm, AdaLayerNorm, FP32LayerNorm
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding
from diffusers.models.activations import FP32SiLU, GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except:
    flash_attn_func = None

logger = logging.get_logger(__name__)


import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import random
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
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



class FlashFP32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kawrgs):
        super(FlashFP32LayerNorm, self).__init__(*args, **kawrgs)
    def forward(self, x):
        origin_dtype = x.dtype
        return layer_norm_fn(
            x.float(), self.weight.float(), 
            self.bias.float() if self.bias is not None else None, eps=self.eps).to(origin_dtype)
    
class PositionGetter3D(object):
    """ return positions of patches """

    def __init__(self,  max_t, max_h, max_w, explicit_uniform_rope=False):
        self.cache_positions = {}
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.explicit_uniform_rope = explicit_uniform_rope
        
    def __call__(self, b, t, h, w, device, training):
        # random.randint is [a, b], but torch.randint is [a, b)
        s_t = random.randint(0, self.max_t-t) if self.explicit_uniform_rope and training else 0
        e_t = s_t + t
        s_h = random.randint(0, self.max_h-h) if self.explicit_uniform_rope and training else 0
        e_h = s_h + h
        s_w = random.randint(0, self.max_w-w) if self.explicit_uniform_rope and training else 0
        e_w = s_w + w
        # print(f'{self.explicit_uniform_rope}, {training}, {b},{s_t},{e_t},{s_h},{e_h},{s_w},{e_w}')
        if not (b,s_t,e_t,s_h,e_h,s_w,e_w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (e_t, e_h, e_w)
            min_poses = (s_t, s_h, s_w)

            self.cache_positions[b,s_t,e_t,s_h,e_h,s_w,e_w] = (poses, min_poses, max_poses)
        pos = self.cache_positions[b,s_t,e_t,s_h,e_h,s_w,e_w]

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

    def get_cos_sin(self, D, seq_start, seq_end, device, interpolation_scale=1):
        if (D, seq_start, seq_start, seq_end, device) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_start, seq_end, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_start, seq_start, seq_end, device] = (cos, sin)
        return self.cache[D, seq_start, seq_start, seq_end, device]

    def forward(self, dim, positions, device):
        """
        input:
            * dim: head_dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (ntokens x batch_size x nheads x dim)
        """
        assert dim % 16 == 0, "number of dimensions should be a multiple of 16"
        D_t = dim // 16 * 4
        D = dim // 16 * 6
        poses, min_poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2 # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D_t, min_poses[0], max_poses[0], device, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, min_poses[1], max_poses[1], device, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, min_poses[2], max_poses[2], device, self.interpolation_scale_w)

        cos_t, sin_t = compute_rope1d(poses[0], cos_t, sin_t)
        cos_y, sin_y = compute_rope1d(poses[1], cos_y, sin_y)
        cos_x, sin_x = compute_rope1d(poses[2], cos_x, sin_x)
        return cos_t, sin_t, cos_y, sin_y, cos_x, sin_x
    
def compute_rope1d(pos1d, cos, sin):
    """
        * pos1d: ntokens x batch_size
    """
    assert pos1d.ndim == 2
    # for (ntokens x batch_size x nheads x dim)
    cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
    sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]

    return cos, sin

def apply_rope1d(tokens, cos, sin):
    """
        * tokens: ntokens x batch_size x nheads x dim
    """
    return (tokens * cos) + (rotate_half(tokens) * sin)
    
def apply_rotary_emb(tokens, video_rotary_emb):
    cos_t, sin_t, cos_y, sin_y, cos_x, sin_x = video_rotary_emb
    # split features into three along the feature dimension, and apply rope1d on each half
    dim = tokens.shape[-1]
    D_t = dim // 16 * 4
    D = dim // 16 * 6
    origin_dtype = tokens.dtype
    t, y, x = torch.split(tokens.float(), [D_t, D, D], dim=-1)
    t = apply_rope1d(t, cos_t, sin_t)
    y = apply_rope1d(y, cos_y, sin_y)
    x = apply_rope1d(x, cos_x, sin_x)
    tokens = torch.cat((t, y, x), dim=-1).to(origin_dtype)
    return tokens

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, timestep_embed_dim, embedding_dim, pooled_projection_dim, time_as_token=False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=timestep_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = None
        if not time_as_token:
            self.timestep_embedder = TimestepEmbedding(in_channels=timestep_embed_dim, time_embed_dim=embedding_dim)
        self.text_embedder = None
        if pooled_projection_dim > 0:
            self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection, dtype):
        timesteps_proj = self.time_proj(timestep)
        if self.timestep_embedder is not None:
            timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=dtype))  # (N, D)
        else:
            timesteps_emb = timesteps_proj.to(dtype=dtype)

        if self.text_embedder is not None:
            pooled_projections = self.text_embedder(pooled_projection) 
            conditioning = timesteps_emb + pooled_projections
        else:
            conditioning = timesteps_emb
        return conditioning

class AdaNorm(AdaLayerNorm):
    def __init__(self, norm_cls='fp32_layer_norm',  **kwargs) -> None:
        super().__init__(**kwargs)
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'fp32_layer_norm':
            self.norm_cls = FP32LayerNorm
        self.norm = self.norm_cls(
            self.norm.normalized_shape, eps=self.norm.eps, elementwise_affine=self.norm.elementwise_affine
            )


class OpenSoraNormZero(nn.Module):
    def __init__(
        self,
        timestep_embed_dim: int, 
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_cls: str = 'fp32_layer_norm', 
    ) -> None:
        super().__init__()
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'fp32_layer_norm':
            self.norm_cls = FP32LayerNorm

        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 6 * embedding_dim, bias=bias)
        self.norm = self.norm_cls(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_enc = self.norm_cls(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[None, :, :] + shift[None, :, :]
        encoder_hidden_states = self.norm_enc(encoder_hidden_states) * (1 + enc_scale)[None, :, :] + enc_shift[None, :, :]

        return hidden_states, encoder_hidden_states, gate[None, :, :], enc_gate[None, :, :]


class OpenSoraNormZero(nn.Module):
    def __init__(
        self,
        timestep_embed_dim: int, 
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_cls: str = 'fp32_layer_norm', 
    ) -> None:
        super().__init__()
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'fp32_layer_norm':
            self.norm_cls = FP32LayerNorm

        self.silu = nn.SiLU()
        self.linear = nn.Linear(timestep_embed_dim, 6 * embedding_dim, bias=bias)
        self.norm = self.norm_cls(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_enc = self.norm_cls(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[None, :, :] + shift[None, :, :]
        encoder_hidden_states = self.norm_enc(encoder_hidden_states) * (1 + enc_scale)[None, :, :] + enc_shift[None, :, :]
        
        return hidden_states, encoder_hidden_states, gate[None, :, :], enc_gate[None, :, :]
    

class OpenSoraAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the OpenSora model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, time_as_x_token=False, time_as_text_token=False):
        self.time_as_x_token = time_as_x_token
        self.time_as_text_token = time_as_text_token
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("OpenSoraAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        frame: int, 
        height: int, 
        width: int, 
        attention_mask: Optional[torch.Tensor] = None,
        video_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, batch_size, _  = hidden_states.shape

        # -----------------------------------------------
        # Step 1, visual token projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        # -----------------------------------------------


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        FA_head_num = attn.heads
        total_frame = frame

        # -----------------------------------------------
        # Step 2, apply qk norm and RoPE
        query = query.view(-1, batch_size, FA_head_num, head_dim)
        key = key.view(-1, batch_size, FA_head_num, head_dim)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        
        if video_rotary_emb is not None:
            if self.time_as_x_token:
                query = torch.cat([query[:1], apply_rotary_emb(query[1:], video_rotary_emb)], dim=0)
                key = torch.cat([key[:1], apply_rotary_emb(key[1:], video_rotary_emb)], dim=0)
            else:
                query = apply_rotary_emb(query, video_rotary_emb)
                key = apply_rotary_emb(key, video_rotary_emb)
        
        query = query.view(-1, batch_size, FA_head_num * head_dim)
        key = key.view(-1, batch_size, FA_head_num * head_dim)

        # -----------------------------------------------


        # -----------------------------------------------
        # Step 3, attention
        if npu_config is not None:
            hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH", head_dim, FA_head_num)
        else:
            # if attention_mask is None and flash_attn_func is not None:
            #     query = rearrange(query, 's b (h d) -> b s h d', h=FA_head_num)
            #     key = rearrange(key, 's b (h d) -> b s h d', h=FA_head_num)
            #     value = rearrange(value, 's b (h d) -> b s h d', h=FA_head_num)
            #     hidden_states = flash_attn_func(
            #         query, key, value, dropout_p=0.0, causal=False
            #     )
            #     hidden_states = rearrange(hidden_states, 'b s h d -> s b (h d)', h=FA_head_num)
            if attention_mask is None:
                query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
                key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
                value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
                hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=FA_head_num)

            else:
                query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
                key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
                value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
                hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=FA_head_num)
        # -----------------------------------------------
        
        # -----------------------------------------------
        # Step 4, proj the attention outputs.
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # -----------------------------------------------

        return hidden_states



class ConvFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, bias=True, activation_fn='geglu', dropout=0.0, final_dropout=False, rep=True):
        super(ConvFeedForward, self).__init__()

        self.rep = rep
        self.bias = bias

        self.hidden_features = hidden_features = inner_dim

        # buffer
        self.buffer = False # False: empty
        self.project_in_weight = None
        self.project_in_bias = None
        self.dwconv_weight = None
        self.dwconv_bias = None
        self.project_out_weight = None
        self.project_out_bias = None

        if rep:
            self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
            self.drop_out = nn.Dropout(dropout)
            self.dwconv = nn.ModuleList([
                nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1, groups=hidden_features, bias=bias),
                nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=hidden_features, bias=bias),
                nn.Conv2d(hidden_features, hidden_features, kernel_size=1, stride=1, padding=0, dilation=1, groups=hidden_features, bias=bias)
            ])

            self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        else:
            self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
            self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)


        self.act_fn = nn.GELU()

        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        self.final_dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()
    def clear_buffer(self):
        self.buffer = False # False: empty
        self.project_in_weight = None
        self.project_in_bias = None
        self.dwconv_weight = None
        self.dwconv_bias = None
        self.project_out_weight = None
        self.project_out_bias = None

    def add_buffer(self):
        self.buffer = True

        w1 = self.project_in.weight.squeeze().detach()
        self.project_in_weight = w1.unsqueeze(-1).unsqueeze(-1)
        self.project_in_bias = None

        if self.bias:
            b1 = self.project_in.bias.detach()
            self.project_in_bias = b1

        self.dwconv_weight = self.dwconv[0].weight.detach()
        self.dwconv_weight[:, :, 1:4, 1:4] += self.dwconv[1].weight.detach()
        self.dwconv_weight[:, :, 2:3, 2:3] += (self.dwconv[2].weight.detach() + 1.) # skip connection
        
        self.dwconv_bias = None
        if self.bias:
            self.dwconv_bias = self.dwconv[0].bias.detach() + self.dwconv[1].bias.detach() + self.dwconv[2].bias.detach()

        w1 = self.project_out.weight.squeeze().detach()
        self.project_out_weight = w1
        self.project_out_bias = None

        if self.bias:
            b1 = self.project_out.bias.detach()
            self.project_out_bias = b1


    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if mode:
            self.clear_buffer() # added: clear rep buffer
        else:
            self.add_buffer() # added: add rep buffer
        for module in self.children():
            module.train(mode)
        return self
    

    def forward(self, x, frame, height, width):
        x = rearrange(x, '(t h w) b c -> (b t) c h w', t=frame, h=height, w=width)

        if (not self.rep) or (self.rep and self.training):
            x = self.project_in(x)
            x = self.act_fn(x)
            x = self.drop_out(x)
            if self.rep:
                out = x
                for module in self.dwconv:
                    out = out + module(x)
            else:
                out = self.dwconv(x)
            x = self.project_out(out)
            x = self.final_dropout(x)
        else: # eval & rep
            x = F.conv2d(x, self.project_in_weight, self.project_in_bias) # project_in
            x = self.act_fn(x)
            x = F.conv2d(x, self.dwconv_weight, self.dwconv_bias, padding=2, groups=self.hidden_features)
            x = F.conv2d(x, self.project_out_weight, self.project_out_bias) # project_out

        x = rearrange(x, '(b t) c h w -> (t h w) b c', t=frame, h=height, w=width)
        return x


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        timestep_embed_dim: int, 
        caption_channels: int, 
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = False,
        attention_out_bias: bool = True,
        context_pre_only: bool = False,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
        norm_cls: str = 'fp32_layer_norm', 
        layerwise_text_mlp: bool = False,
        time_as_x_token: bool = False,
        time_as_text_token: bool = False,
        sandwich_norm: bool = False, 
        conv_ffn: bool = False,
        prenorm: bool = True, 
    ):
        super().__init__()
        self.sandwich_norm = sandwich_norm
        self.time_as_x_token = time_as_x_token
        self.time_as_text_token = time_as_text_token
        self.time_as_token = time_as_x_token or time_as_text_token
        self.prenorm = prenorm

        self.attention_head_dim = attention_head_dim
        self.layerwise_text_mlp = layerwise_text_mlp
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'fp32_layer_norm':
            self.norm_cls = FP32LayerNorm

        if self.time_as_token:
            self.map_time_to_token = nn.Sequential(
                nn.Linear(in_features=timestep_embed_dim, out_features=dim, bias=True), 
                FP32SiLU(), 
                nn.Linear(in_features=dim, out_features=caption_channels if time_as_text_token else dim, bias=True)
            )
        else:
            self.silu = nn.SiLU()
            self.linear = nn.Linear(timestep_embed_dim, 9 * dim, bias=False)

        # 1. Self-Attn
        self.norm1 = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None, 
            dim_head=attention_head_dim, 
            heads=num_attention_heads,
            context_pre_only=None,
            qk_norm=norm_cls,
            eps=norm_eps,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=OpenSoraAttnProcessor2_0(time_as_x_token=time_as_x_token, time_as_text_token=False),
        )
        if self.sandwich_norm:
            self.sandwich_norm1 = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        # 3. Cross-Attn
        if self.layerwise_text_mlp:
            self.text_norm_mlp = nn.Sequential(
                self.norm_cls(caption_channels, eps=norm_eps, elementwise_affine=norm_elementwise_affine), 
                nn.Linear(caption_channels, caption_channels, bias=True), 
                FP32SiLU()
            )
        self.norm2 = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=caption_channels,
            added_kv_proj_dim=None, 
            dim_head=attention_head_dim, 
            heads=num_attention_heads,
            context_pre_only=None,
            qk_norm=norm_cls,
            eps=norm_eps,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=OpenSoraAttnProcessor2_0(time_as_x_token=False, time_as_text_token=False),
        )
        if self.sandwich_norm:
            self.sandwich_norm2 = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        # 3. Feed-forward
        self.norm3 = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        ffn_cls = ConvFeedForward if conv_ffn else FeedForward
        ff_inner_dim = int(2.5 * dim) if conv_ffn else 4 * dim
        self.ff = ffn_cls(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        ) 
        if self.sandwich_norm:
            self.sandwich_norm3 = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    
    def _prenorm_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        video_rotary_emb = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> torch.FloatTensor:
        B = embedded_timestep.shape[0]
        device = hidden_states.device
        # 0. Prepare module
        if self.time_as_token:
            embedded_timestep = self.map_time_to_token(embedded_timestep)
        else:
            shift, scale, gate, crs_shift, crs_scale, crs_gate, ffn_shift, ffn_scale, ffn_gate = \
                self.linear(self.silu(embedded_timestep)).chunk(9, dim=1)

        if self.time_as_token:
            if self.time_as_x_token:
                hidden_states = torch.cat([embedded_timestep.unsqueeze(0), hidden_states], dim=0)
                if attention_mask is not None:
                    attention_mask = torch.cat([torch.zeros(B, 1, 1, 1, device=device), attention_mask], dim=-1)
            norm_hidden_states = self.norm1(hidden_states)
        else:
            # norm & scale & shift
            norm_hidden_states = self.norm1(hidden_states) * (1 + scale)[None, :, :] + shift[None, :, :]

        # 1. Self-Attention
        attn_hidden_states = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            frame=frame, 
            height=height, 
            width=width, 
            attention_mask=attention_mask, 
            video_rotary_emb=video_rotary_emb, 
        )
        if self.time_as_token:
            if self.sandwich_norm:
                attn_hidden_states = self.sandwich_norm1(attn_hidden_states)
            hidden_states = hidden_states + attn_hidden_states
        else:
            # residual & gate
            if self.sandwich_norm:
                hidden_states = hidden_states + self.sandwich_norm1(gate[None, :, :] * attn_hidden_states)
            else:
                hidden_states = hidden_states + gate[None, :, :] * attn_hidden_states
        

        if self.time_as_token:
            norm_hidden_states = self.norm2(hidden_states)
            if self.time_as_text_token:
                encoder_hidden_states = torch.cat([embedded_timestep.unsqueeze(0), encoder_hidden_states], dim=0)
                if encoder_attention_mask is not None:
                    encoder_attention_mask = torch.cat([torch.zeros(B, 1, 1, 1, device=device), encoder_attention_mask], dim=-1)
        else:
            # norm & scale & shift
            norm_hidden_states = self.norm2(hidden_states) * (1 + crs_scale)[None, :, :] + crs_shift[None, :, :]

        # 2. Cross-Attention
        if self.layerwise_text_mlp:
            encoder_hidden_states = self.text_norm_mlp(encoder_hidden_states)
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            frame=frame, 
            height=height, 
            width=width, 
            attention_mask=encoder_attention_mask, 
            video_rotary_emb=None, 
        )
        if self.time_as_token:
            if self.sandwich_norm:
                attn_hidden_states = self.sandwich_norm2(attn_hidden_states)
            hidden_states = hidden_states + attn_hidden_states
        else:
            # residual & gate
            if self.sandwich_norm:
                hidden_states = hidden_states + self.sandwich_norm2(crs_gate[None, :, :] * attn_hidden_states)
            else:
                hidden_states = hidden_states + crs_gate[None, :, :] * attn_hidden_states


        
        if self.time_as_token:
            if self.time_as_x_token:
                hidden_states = hidden_states[1:]
            norm_hidden_states = self.norm3(hidden_states)
        else:
            # norm & scale & shift
            norm_hidden_states = self.norm3(hidden_states) * (1 + ffn_scale)[None, :, :] + ffn_shift[None, :, :]

        # 3. Share Feed-Forward
        ff_output = self.ff(norm_hidden_states, frame=frame, height=height, width=width)
        if self.time_as_token:
            if self.sandwich_norm:
                ff_output = self.sandwich_norm3(ff_output)
            hidden_states = hidden_states + ff_output
        else:
            # residual & gate
            if self.sandwich_norm:
                hidden_states = hidden_states + self.sandwich_norm3(ffn_gate[None, :, :] * ff_output)
            else:
                hidden_states = hidden_states + ffn_gate[None, :, :] * ff_output

        return hidden_states



    def _postnorm_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        video_rotary_emb = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> torch.FloatTensor:
        assert not self.sandwich_norm
        B = embedded_timestep.shape[0]
        device = hidden_states.device
        # 0. Prepare module
        if self.time_as_token:
            embedded_timestep = self.map_time_to_token(embedded_timestep)
        else:
            shift, scale, gate, crs_shift, crs_scale, crs_gate, ffn_shift, ffn_scale, ffn_gate = \
                self.linear(self.silu(embedded_timestep)).chunk(9, dim=1)

        if self.time_as_token:
            if self.time_as_x_token:
                hidden_states = torch.cat([embedded_timestep.unsqueeze(0), hidden_states], dim=0)
                if attention_mask is not None:
                    attention_mask = torch.cat([torch.zeros(B, 1, 1, 1, device=device), attention_mask], dim=-1)
            res = hidden_states
            
        else:
            res = hidden_states
            # scale & shift
            hidden_states = hidden_states * (1 + scale)[None, :, :] + shift[None, :, :]

        # 1. Self-Attention
        attn_hidden_states = self.attn1(
            hidden_states,
            encoder_hidden_states=None,
            frame=frame, 
            height=height, 
            width=width, 
            attention_mask=attention_mask, 
            video_rotary_emb=video_rotary_emb, 
        )
        if self.time_as_token:
            hidden_states = res + attn_hidden_states
        else:
            # residual & gate
            hidden_states = res + gate[None, :, :] * attn_hidden_states
        hidden_states = self.norm1(hidden_states)
        


        res = hidden_states
        if self.time_as_token:
            if self.time_as_text_token:
                encoder_hidden_states = torch.cat([embedded_timestep.unsqueeze(0), encoder_hidden_states], dim=0)
                if encoder_attention_mask is not None:
                    encoder_attention_mask = torch.cat([torch.zeros(B, 1, 1, 1, device=device), encoder_attention_mask], dim=-1)
        else:
            # scale & shift
            hidden_states = hidden_states * (1 + crs_scale)[None, :, :] + crs_shift[None, :, :]

        # 2. Cross-Attention
        if self.layerwise_text_mlp:
            encoder_hidden_states = self.text_norm_mlp(encoder_hidden_states)
        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            frame=frame, 
            height=height, 
            width=width, 
            attention_mask=encoder_attention_mask, 
            video_rotary_emb=None, 
        )

        
        if self.time_as_token:
            hidden_states = res + attn_hidden_states
        else:
            # residual & gate
            hidden_states = res + crs_gate[None, :, :] * attn_hidden_states
        hidden_states = self.norm2(hidden_states)




        if self.time_as_token:
            if self.time_as_x_token:
                hidden_states = hidden_states[1:]
            res = hidden_states
        else:
            res = hidden_states
            # scale & shift
            hidden_states = hidden_states * (1 + ffn_scale)[None, :, :] + ffn_shift[None, :, :]


        # 3. Share Feed-Forward
        ff_output = self.ff(hidden_states, frame=frame, height=height, width=width)

        if self.time_as_token:
            hidden_states = res + ff_output
        else:
            # residual & gate
            hidden_states = res + ffn_gate[None, :, :] * ff_output
        hidden_states = self.norm3(hidden_states)
        
        return hidden_states


    def forward(self, *args, **kwargs):
        if self.prenorm:
            print('prenorm')
            return self._prenorm_forward(*args, **kwargs)
        else:
            print('postnorm')
            return self._postnorm_forward(*args, **kwargs)