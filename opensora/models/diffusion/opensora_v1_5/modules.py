
import torch
from torch import nn
from typing import Optional, Tuple
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from torch.nn import functional as F
from diffusers.models.normalization import RMSNorm, AdaLayerNorm
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding

logger = logging.get_logger(__name__)


import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
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


class PositionGetter3D(object):
    """ return positions of patches """

    def __init__(self, ):
        self.cache_positions = {}
        
    def __call__(self, b, t, h, w, device):
        if not (b,t,h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]

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

    def forward(self, dim, positions, device, dtype):
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
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2 # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D_t, max_poses[0] + 1, device, dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, device, dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, device, dtype, self.interpolation_scale_w)
        return poses, cos_t, sin_t, cos_y, sin_y, cos_x, sin_x
    
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope1d(tokens, pos1d, cos, sin):
    """
        * tokens: ntokens x batch_size x nheads x dim
        * pos1d: ntokens x batch_size
    """
    assert pos1d.ndim == 2
    # for (ntokens x batch_size x nheads x dim)
    cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
    sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]

    return (tokens * cos) + (rotate_half(tokens) * sin)
    
def apply_rotary_emb(tokens, video_rotary_emb):
    poses, cos_t, sin_t, cos_y, sin_y, cos_x, sin_x = video_rotary_emb
    # split features into three along the feature dimension, and apply rope1d on each half
    dim = tokens.shape[-1]
    D_t = dim // 16 * 4
    D = dim // 16 * 6
    t, y, x = torch.split(tokens, [D_t, D, D], dim=-1)
    t = apply_rope1d(t, poses[0], cos_t, sin_t)
    y = apply_rope1d(y, poses[1], cos_y, sin_y)
    x = apply_rope1d(x, poses[2], cos_x, sin_x)
    tokens = torch.cat((t, y, x), dim=-1)
    return tokens

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

class AdaNorm(AdaLayerNorm):
    def __init__(self, norm_cls='rms_norm',  **kwargs) -> None:
        super().__init__(**kwargs)
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm
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
        norm_cls: str = 'rms_norm', 
    ) -> None:
        super().__init__()
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm

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
        # shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        # hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        # encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        # return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]



class OpenSoraAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the OpenSora model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, sparse1d, sparse_n, sparse_group):
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("OpenSoraAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

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
        text_seq_length, batch_size, _  = encoder_hidden_states.shape

        # -----------------------------------------------
        # Step 1, visual token projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        # print(f'hidden_states query ', 
        #         f'max {query.max()}, min {query.min()}, mean {query.mean()}, std {query.std()}')
        # print(f'hidden_states key ', 
        #         f'max {key.max()}, min {key.min()}, mean {key.mean()}, std {key.std()}')
        # print(f'hidden_states value ', 
        #         f'max {value.max()}, min {value.min()}, mean {value.mean()}, std {value.std()}')
        # -----------------------------------------------

        # -----------------------------------------------
        # Step 2, text token projection
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        # print(f'encoder_hidden_states_query_proj ', 
        #         f'max {encoder_hidden_states_query_proj.max()}, min {encoder_hidden_states_query_proj.min()}, mean {encoder_hidden_states_query_proj.mean()}, std {encoder_hidden_states_query_proj.std()}')
        # print(f'encoder_hidden_states_key_proj ', 
        #         f'max {encoder_hidden_states_key_proj.max()}, min {encoder_hidden_states_key_proj.min()}, mean {encoder_hidden_states_key_proj.mean()}, std {encoder_hidden_states_key_proj.std()}')
        # print(f'encoder_hidden_states_value_proj ', 
        #         f'max {encoder_hidden_states_value_proj.max()}, min {encoder_hidden_states_value_proj.min()}, mean {encoder_hidden_states_value_proj.mean()}, std {encoder_hidden_states_value_proj.std()}')
        # -----------------------------------------------

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        FA_head_num = attn.heads
        total_frame = frame

        # -----------------------------------------------
        # Step 3, apply qk norm and RoPE
        query = query.view(-1, batch_size, FA_head_num, head_dim)
        key = key.view(-1, batch_size, FA_head_num, head_dim)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        # print(f'norm_q ', 
        #         f'max {query.max()}, min {query.min()}, mean {query.mean()}, std {query.std()}')
        # print(f'norm_k ', 
        #         f'max {key.max()}, min {key.min()}, mean {key.mean()}, std {key.std()}')

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(-1, batch_size, FA_head_num, head_dim)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(-1, batch_size, FA_head_num, head_dim)
        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        # print(f'norm_added_q ', 
        #         f'max {encoder_hidden_states_query_proj.max()}, min {encoder_hidden_states_query_proj.min()}, mean {encoder_hidden_states_query_proj.mean()}, std {encoder_hidden_states_query_proj.std()}')
        # print(f'norm_added_k ', 
        #         f'max {encoder_hidden_states_key_proj.max()}, min {encoder_hidden_states_key_proj.min()}, mean {encoder_hidden_states_key_proj.mean()}, std {encoder_hidden_states_key_proj.std()}')

        query = apply_rotary_emb(query, video_rotary_emb)
        key = apply_rotary_emb(key, video_rotary_emb)
        # print(f'apply_rotary_emb query ', 
        #         f'max {query.max()}, min {query.min()}, mean {query.mean()}, std {query.std()}')
        # print(f'apply_rotary_emb key ', 
        #         f'max {key.max()}, min {key.min()}, mean {key.mean()}, std {key.std()}')

        query = query.view(-1, batch_size, FA_head_num * head_dim)
        key = key.view(-1, batch_size, FA_head_num * head_dim)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(-1, batch_size, FA_head_num * head_dim)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(-1, batch_size, FA_head_num * head_dim)
        # -----------------------------------------------

        
        # -----------------------------------------------
        # Step 4, sparse token
        # print(f'dense v shape: query {query.shape}, key {key.shape}, value {value.shape}')
        # print(f'dense t shape: encoder_hidden_states_query_proj {encoder_hidden_states_query_proj.shape}, encoder_hidden_states_key_proj {encoder_hidden_states_key_proj.shape}, encoder_hidden_states_value_proj {encoder_hidden_states_value_proj.shape}')
        if self.sparse1d:
            query, pad_len = self._sparse_1d(query, total_frame, height, width)
            key, pad_len = self._sparse_1d(key, total_frame, height, width)
            value, pad_len = self._sparse_1d(value, total_frame, height, width)
            encoder_hidden_states_query_proj = self._sparse_1d_enc(encoder_hidden_states_query_proj)
            encoder_hidden_states_key_proj = self._sparse_1d_enc(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self._sparse_1d_enc(encoder_hidden_states_value_proj)
            # print(f'sparse v shape: query {query.shape}, key {key.shape}, value {value.shape}')
            # print(f'sparse t shape: encoder_hidden_states_query_proj {encoder_hidden_states_query_proj.shape}, encoder_hidden_states_key_proj {encoder_hidden_states_key_proj.shape}, encoder_hidden_states_value_proj {encoder_hidden_states_value_proj.shape}')
        
        # -----------------------------------------------


        # -----------------------------------------------
        # Step 5, attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=0)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=0)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=0)
        # print(f'cat shape: query {query.shape}, key {key.shape}, value {value.shape}')

        if npu_config is not None:
            hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH", head_dim, FA_head_num)
        else:
            query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
            key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
            value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=FA_head_num)
        # -----------------------------------------------

        # print(f'scaled_dot_product_attention hidden_states ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')

        # -----------------------------------------------
        # Step 6, split->reverse sparse->proj the attention outputs.
        hidden_states, encoder_hidden_states = hidden_states.split(
            [hidden_states.size(0) - text_seq_length, text_seq_length], dim=0
        )
        # print(f'split v shape: hidden_states {hidden_states.shape}')
        # print(f'split t shape: encoder_hidden_states {encoder_hidden_states.shape}')
        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frame, height, width, pad_len)
            encoder_hidden_states = self._reverse_sparse_1d_enc(encoder_hidden_states)
            # print(f'split v shape: hidden_states {hidden_states.shape}')
            # print(f'split t shape: encoder_hidden_states {encoder_hidden_states.shape}')

        # print(f'before to_out hidden_states ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'before to_out encoder_hidden_states ', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        
        # hidden_states = attn.out_prenorm(hidden_states)
        # print(f'hidden_states out_prenorm ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            # encoder_hidden_states = attn.add_out_prenorm(encoder_hidden_states)
            # print(f'encoder_hidden_states out_prenorm ', 
            #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
            # linear proj for text feature
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        # -----------------------------------------------

        # print(f'to_out hidden_states ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'to_out encoder_hidden_states ', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        # import sys;sys.exit()
        return hidden_states, encoder_hidden_states

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        timestep_embed_dim: int, 
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
        sparse1d: bool = False,
        sparse_n: int = 2,
        sparse_group: bool = False,
        norm_cls: str = 'rms_norm', 
    ):
        super().__init__()
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.attention_head_dim = attention_head_dim
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm

        # 1. Self-Attn
        self.norm1 = OpenSoraNormZero(
            timestep_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, norm_cls=norm_cls
            )
        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim, 
            dim_head=attention_head_dim, 
            heads=num_attention_heads,
            context_pre_only=context_pre_only,
            qk_norm=norm_cls,
            eps=norm_eps,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=OpenSoraAttnProcessor2_0(sparse1d, sparse_n, sparse_group),
        )
        self.attn1.norm_added_q = self.norm_cls(attention_head_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn1.norm_added_k = self.norm_cls(attention_head_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        # self.attn1.out_prenorm = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        # if not context_pre_only:
        #     self.attn1.add_out_prenorm = self.norm_cls(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        # 2. Feed-forward
        self.norm2 = OpenSoraNormZero(
            timestep_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, norm_cls=norm_cls
            )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> torch.FloatTensor:
        
        # 0. Prepare rope embedding
        vis_seq_length, batch_size = hidden_states.size(0), hidden_states.size(1)
        pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=hidden_states.device)
        video_rotary_emb = self.rope(self.attention_head_dim, pos_thw, hidden_states.device, hidden_states.dtype)

        # print(f'hidden_states input', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'encoder_hidden_states input', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        # 1. Self-Attention
        # norm & scale & shift
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, embedded_timestep
            )
        # print(f'norm_hidden_states norm1', 
        #         f'max {norm_hidden_states.max()}, min {norm_hidden_states.min()}, mean {norm_hidden_states.mean()}, std {norm_hidden_states.std()}')
        # print(f'norm_encoder_hidden_states norm1', 
        #         f'max {norm_encoder_hidden_states.max()}, min {norm_encoder_hidden_states.min()}, mean {norm_encoder_hidden_states.mean()}, std {norm_encoder_hidden_states.std()}')
        # attn
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            frame=frame, 
            height=height, 
            width=width, 
            attention_mask=attention_mask, 
            video_rotary_emb=video_rotary_emb, 
        )
        # print(f'attn_hidden_states attn1', 
        #         f'max {attn_hidden_states.max()}, min {attn_hidden_states.min()}, mean {attn_hidden_states.mean()}, std {attn_hidden_states.std()}')
        # print(f'attn_encoder_hidden_states attn1', 
        #         f'max {attn_encoder_hidden_states.max()}, min {attn_encoder_hidden_states.min()}, mean {attn_encoder_hidden_states.mean()}, std {attn_encoder_hidden_states.std()}')
        # residual & gate
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # print(f'(gate_msa * attn_hidden_states) gate', 
        #         f'max {(gate_msa * attn_hidden_states).max()}, min {(gate_msa * attn_hidden_states).min()}, mean {(gate_msa * attn_hidden_states).mean()}, std {(gate_msa * attn_hidden_states).std()}')
        # print(f'(enc_gate_msa * attn_encoder_hidden_states) gate', 
        #         f'max {(enc_gate_msa * attn_encoder_hidden_states).max()}, min {(enc_gate_msa * attn_encoder_hidden_states).min()}, mean {(enc_gate_msa * attn_encoder_hidden_states).mean()}, std {(enc_gate_msa * attn_encoder_hidden_states).std()}')
        
        # print(f'hidden_states residual & gate', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'encoder_hidden_states residual & gate', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        # 1. Share Feed-Forward
        # norm & scale & shift
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, embedded_timestep
        )
        # print(f'norm_hidden_states norm & scale & shift', 
        #         f'max {norm_hidden_states.max()}, min {norm_hidden_states.min()}, mean {norm_hidden_states.mean()}, std {norm_hidden_states.std()}')
        # print(f'norm_encoder_hidden_states norm & scale & shift', 
        #         f'max {norm_encoder_hidden_states.max()}, min {norm_encoder_hidden_states.min()}, mean {norm_encoder_hidden_states.mean()}, std {norm_encoder_hidden_states.std()}')
        # ffn
        norm_hidden_states = torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=0)
        ff_output = self.ff(norm_hidden_states)
        # print(f'ff_output ff', 
        #         f'max {ff_output.max()}, min {ff_output.min()}, mean {ff_output.mean()}, std {ff_output.std()}')
        # residual & gate
        hidden_states = hidden_states + gate_ff * ff_output[:vis_seq_length]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[vis_seq_length:]
        # print(f'hidden_states ffn residual & gate', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'encoder_hidden_states ffn residual & gate', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')

        return hidden_states, encoder_hidden_states
