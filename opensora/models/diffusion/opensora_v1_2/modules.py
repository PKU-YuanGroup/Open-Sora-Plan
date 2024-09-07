from einops import rearrange
from torch import nn
import torch
import numpy as np

from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from diffusers.utils.torch_utils import maybe_allow_in_graph
from typing import Any, Dict, Optional
import re
import torch
import torch.nn.functional as F
from torch import nn
import diffusers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward, GatedSelfAttentionDense
from diffusers.models.attention_processor import Attention as Attention_
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, Timesteps, TimestepEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from .rope import PositionGetter3D, RoPE3D
try:
    import torch_npu
    from opensora.npu_config import npu_config, set_run_dtype
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
    from opensora.acceleration.communications import all_to_all_SBH
except:
    torch_npu = None
    npu_config = None
    set_run_dtype = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info
    from opensora.utils.communications import all_to_all_SBH
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
    
class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with video"""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)
        latent = rearrange(latent, '(b t) c h w -> b (t h w) c', b=b)
        return latent

class Attention(Attention_):
    def __init__(
            self, interpolation_scale_thw, sparse1d, sparse2d, sparse_n, 
            sparse_group, is_cross_attn, **kwags
            ):
        processor = OpenSoraAttnProcessor2_0(
            interpolation_scale_thw=interpolation_scale_thw, sparse1d=sparse1d, sparse2d=sparse2d, sparse_n=sparse_n, 
            sparse_group=sparse_group, is_cross_attn=is_cross_attn
            )
        super().__init__(processor=processor, **kwags)
        
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
            head_size = head_size // nccl_info.world_size
        
        if attention_mask is None:  # b 1 t*h*w in sa, b 1 l in ca, target_length 0
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
                 sparse1d=False, sparse2d=False, sparse_n=2, sparse_group=False, is_cross_attn=True):
        self.sparse1d = sparse1d
        self.sparse2d = sparse2d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        
        self._init_rope(interpolation_scale_thw)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        assert not (self.sparse1d and self.sparse2d)

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()
    
    def _sparse_1d(self, x, attention_mask, frame, height, width):
        """
        require the shape of (batch_size x nheads x ntokens x dim)
        attention_mask: b nheads 1 thw
        """
        l = x.shape[-2]
        assert l == frame*height*width
        if torch_npu is not None and attention_mask is not None:
            assert attention_mask.ndim == 3 and attention_mask.shape[1] == 1
            attention_mask = attention_mask.unsqueeze(1)
        assert attention_mask is None or attention_mask.shape[2] == 1
        pad_len = 0
        if l % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - l % (self.sparse_n * self.sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = F.pad(attention_mask, (0, pad_len, 0, 0), value=-9980.0)
        if not self.sparse_group:
            x = rearrange(x, 'b h (g k) d -> (k b) h g d', k=self.sparse_n)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = rearrange(attention_mask, 'b h 1 (g k) -> (k b) h 1 g', k=self.sparse_n).contiguous()
        else:
            x = rearrange(x, 'b h (n m k) d -> (m b) h (n k) d', m=self.sparse_n, k=self.sparse_n)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = rearrange(attention_mask, 'b h 1 (n m k) -> (m b) h 1 (n k)', m=self.sparse_n, k=self.sparse_n)
        if self.is_cross_attn:
            attention_mask = attention_mask.repeat(self.sparse_n, 1, 1, 1)
        return x, attention_mask, pad_len

    def _sparse_1d_on_npu(self, x, attention_mask, frame, height, width):
        """
        require the shape of (batch_size x ntokens x nheads x dim)
        attention_mask: b nheads 1 thw
        """
        l = x.shape[1]
        assert l == frame*height*width
        if torch_npu is not None and attention_mask is not None:
            assert attention_mask.ndim == 3 and attention_mask.shape[1] == 1
            attention_mask = attention_mask.unsqueeze(1)
        assert attention_mask is None or attention_mask.shape[2] == 1
        pad_len = 0
        if l % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - l % (self.sparse_n * self.sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = F.pad(attention_mask, (0, pad_len, 0, 0), value=-9980.0)
        if not self.sparse_group:
            x = rearrange(x, 'b (g k) h d -> (b k) g h d', k=self.sparse_n)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = rearrange(attention_mask, 'b h 1 (g k) -> (b k) h 1 g', k=self.sparse_n).contiguous()
        else:
            x = rearrange(x, 'b (n m k) h d -> (b m) (n k) h d', m=self.sparse_n, k=self.sparse_n)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = rearrange(attention_mask, 'b h 1 (n m k) -> (b m) h 1 (n k)', m=self.sparse_n, k=self.sparse_n)
        if self.is_cross_attn:
            attention_mask = repeat(attention_mask, 'b h 1 s -> (b k) h 1 s', k=self.sparse_n)
        return x, attention_mask, pad_len
    
    def _reverse_sparse_1d(self, x, frame, height, width, pad_len):
        """
        require the shape of (batch_size x nheads x ntokens x dim)
        """
        # import ipdb;ipdb.set_trace()
        assert x.shape[2] == (frame*height*width+pad_len) // self.sparse_n
        if not self.sparse_group:
            x = rearrange(x, '(k b) h g d -> b h (g k) d', k=self.sparse_n)
        else:
            x = rearrange(x, '(m b) h (n k) d -> b h (n m k) d', m=self.sparse_n, k=self.sparse_n)
        x = x[:, :, :frame*height*width, :]
        return x

    def _reverse_sparse_1d_on_npu(self, x, frame, height, width, pad_len):
        """
        require the shape of (batch_size x ntokens x nheads x dim)
        """
        assert x.shape[1] == (frame * height * width + pad_len) // self.sparse_n
        if not self.sparse_group:
            x = rearrange(x, '(b k) g h d -> b (g k) h d', k=self.sparse_n)
        else:
            x = rearrange(x, '(b m) (n k) h d -> b (n m k) h d', m=self.sparse_n, k=self.sparse_n)
        x = x[:, :frame*height*width, :, :]
        return x
    
    def _sparse_1d_kv(self, x):
        """
        require the shape of (batch_size x nheads x ntokens x dim)
        """
        x = repeat(x, 'b h s d -> (k b) h s d', k=self.sparse_n)
        return x

    def _sparse_1d_kv_on_npu(self, x):
        """
        require the shape of (batch_size x ntokens x nheads x dim)
        """
        x = repeat(x, 'b h s d -> (b k) h s d', k=self.sparse_n)
        return x
    
    def _sparse_2d(self, x, attention_mask, frame, height, width):
        """
        require the shape of (batch_size x nheads x ntokens x dim)
        attention_mask: b nheads 1 thw
        """
        d = x.shape[-1]
        x = rearrange(x, 'b h (T H W) d -> b h T H W d', T=frame, H=height, W=width)
        if torch_npu is not None and attention_mask is not None:
            assert attention_mask.ndim == 3 and attention_mask.shape[1] == 1
            attention_mask = attention_mask.unsqueeze(1)
        if attention_mask is not None and not self.is_cross_attn:
            attention_mask = rearrange(attention_mask, 'b h 1 (T H W) -> b h T H W', T=frame, H=height, W=width)
        pad_height = self.sparse_n*self.sparse_n - height % (self.sparse_n*self.sparse_n)
        pad_width = self.sparse_n*self.sparse_n - width % (self.sparse_n*self.sparse_n)
        if pad_height != 0 or pad_width != 0:
            x = rearrange(x, 'b h T H W d -> b (h d) T H W')
            x = F.pad(x, (0, pad_width, 0, pad_height, 0, 0))
            x = rearrange(x, 'b (h d) T H W -> b h T H W d', d=d)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = F.pad(attention_mask, (0, pad_width, 0, pad_height, 0, 0), value=-9500.0)

        if not self.sparse_group:
            x = rearrange(x, 'b h t (g1 k1) (g2 k2) d -> (k1 k2 b) h (t g1 g2) d', 
                          k1=self.sparse_n, k2=self.sparse_n)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = rearrange(attention_mask, 'b h t (g1 k1) (g2 k2) -> (k1 k2 b) h 1 (t g1 g2)', 
                          k1=self.sparse_n, k2=self.sparse_n).contiguous()
        else:
            x = rearrange(x, 'b h t (n1 m1 k1) (n2 m2 k2) d -> (m1 m2 b) h (t n1 n2 k1 k2) d', 
                          m1=self.sparse_n, k1=self.sparse_n, m2=self.sparse_n, k2=self.sparse_n)
            if attention_mask is not None and not self.is_cross_attn:
                attention_mask = rearrange(attention_mask, 'b h t (n1 m1 k1) (n2 m2 k2) -> (m1 m2 b) h 1 (t n1 n2 k1 k2)', 
                          m1=self.sparse_n, k1=self.sparse_n, m2=self.sparse_n, k2=self.sparse_n)
        
        if self.is_cross_attn:
            attention_mask = attention_mask.repeat(self.sparse_n*self.sparse_n, 1, 1, 1)
        return x, attention_mask, pad_height, pad_width
    
    def _reverse_sparse_2d(self, x, frame, height, width, pad_height, pad_width):
        """
        require the shape of (batch_size x nheads x ntokens x dim)
        """
        assert x.shape[2] == frame*(height+pad_height)*(width+pad_width)//self.sparse_n//self.sparse_n
        if not self.sparse_group:
            x = rearrange(x, '(k1 k2 b) h (t g1 g2) d -> b h t (g1 k1) (g2 k2) d', 
                          k1=self.sparse_n, k2=self.sparse_n, 
                          g1=(height+pad_height)//self.sparse_n, g2=(width+pad_width)//self.sparse_n)
        else:
            x = rearrange(x, '(m1 m2 b) h (t n1 n2 k1 k2) d -> b h t (n1 m1 k1) (n2 m2 k2) d', 
                          m1=self.sparse_n, k1=self.sparse_n, m2=self.sparse_n, k2=self.sparse_n, 
                          n1=(height+pad_height)//self.sparse_n//self.sparse_n, n2=(width+pad_width)//self.sparse_n//self.sparse_n)
        x = x[:, :, :, :height, :width, :]
        x = rearrange(x, 'b h T H W d -> b h (T H W) d')
        return x
    
    def _sparse_2d_kv(self, x):
        """
        require the shape of (batch_size x nheads x ntokens x dim)
        """
        x = repeat(x, 'b h s d -> (k1 k2 b) h s d', k1=self.sparse_n, k2=self.sparse_n)
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
        
        if get_sequence_parallel_state():
            if npu_config is not None:
                sequence_length, batch_size, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
            else:
                sequence_length, batch_size, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

        if attention_mask is not None:
            if npu_config is None:
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                if get_sequence_parallel_state():
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length * nccl_info.world_size, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads // nccl_info.world_size, -1, attention_mask.shape[-1])
                else:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        if npu_config is not None and npu_config.on_npu:
            if get_sequence_parallel_state():
                query = query.view(-1, attn.heads, head_dim)  # [s // sp, b, h * d] -> [s // sp * b, h, d]
                key = key.view(-1, attn.heads, head_dim)
                value = value.view(-1, attn.heads, head_dim)
                
                h_size = attn.heads * head_dim
                sp_size = hccl_info.world_size
                h_size_sp = h_size // sp_size
                # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
                query = all_to_all_SBH(query, scatter_dim=1, gather_dim=0).view(-1, batch_size, h_size_sp)
                key = all_to_all_SBH(key, scatter_dim=1, gather_dim=0).view(-1, batch_size, h_size_sp)
                value = all_to_all_SBH(value, scatter_dim=1, gather_dim=0).view(-1, batch_size, h_size_sp)
                
                query = query.view(-1, batch_size, attn.heads // sp_size, head_dim)
                key = key.view(-1, batch_size, attn.heads // sp_size, head_dim)

                if not self.is_cross_attn:
                    # require the shape of (batch_size x nheads x ntokens x dim)
                    pos_thw = self.position_getter(batch_size, t=frame * sp_size, h=height, w=width, device=query.device)
                    query = self.rope(query, pos_thw)
                    key = self.rope(key, pos_thw)

                query = query.view(-1, batch_size, h_size_sp)
                key = key.view(-1, batch_size, h_size_sp)
                value = value.view(-1, batch_size, h_size_sp)
                hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH",
                                                         head_dim, attn.heads // sp_size)

                hidden_states = hidden_states.view(-1, attn.heads // sp_size, head_dim)

                # [s * b, h // sp, d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
                hidden_states = all_to_all_SBH(hidden_states, scatter_dim=0, gather_dim=1).view(-1, batch_size, h_size)
            else:
                if npu_config.enable_FA and query.dtype == torch.float32:
                    dtype = torch.bfloat16
                else:
                    dtype = None

                query = query.reshape(batch_size, -1, attn.heads, head_dim)
                key = key.reshape(batch_size, -1, attn.heads, head_dim)

                if not self.is_cross_attn:
                    # require the shape of (batch_size x ntokens x nheads x dim)
                    pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
                    query = self.rope(query, pos_thw)
                    key = self.rope(key, pos_thw)

                value = value.reshape(batch_size, -1, attn.heads, head_dim)
                
                if self.sparse1d:
                    query, attention_mask, pad_len = self._sparse_1d_on_npu(query, attention_mask, frame, height, width)
                    
                    if self.is_cross_attn:
                        key = self._sparse_1d_kv_on_npu(key)
                        value = self._sparse_1d_kv_on_npu(value)
                    else:
                        key, _, pad_len = self._sparse_1d_on_npu(key, None, frame, height, width)
                        value, _, pad_len = self._sparse_1d_on_npu(value, None, frame, height, width)

                elif self.sparse2d:
                    
                    query, attention_mask, pad_height, pad_width = self._sparse_2d(query, attention_mask, frame, height, width)
                    if self.is_cross_attn:
                        key = self._sparse_2d_kv(key)
                        value = self._sparse_2d_kv(value)
                    else:
                        key, _, pad_height, pad_width = self._sparse_2d(key, None, frame, height, width)
                        value, _, pad_height, pad_width = self._sparse_2d(value, None, frame, height, width)

                query = query.reshape(query.shape[0], query.shape[1], -1)
                key = key.reshape(key.shape[0], key.shape[1], -1)
                value = value.reshape(value.shape[0], value.shape[1], -1)

                if npu_config is not None and attention_mask is not None:
                    if self.sparse1d or self.sparse2d:
                        assert attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1 and attention_mask.ndim == 4
                        attention_mask = attention_mask.squeeze(1) #  b 1 l
                    else:
                        assert attention_mask.shape[1] == 1 and attention_mask.ndim == 3
                    if self.is_cross_attn:
                        attention_mask = npu_config.get_attention_mask(attention_mask, query.shape[1])
                        attention_mask = attention_mask.reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
                    else:
                        attention_mask = npu_config.get_attention_mask(attention_mask, attention_mask.shape[-1])
                        attention_mask = attention_mask.reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
                with set_run_dtype(query, dtype):
                    query, key, value = npu_config.set_current_run_dtype([query, key, value])
                    hidden_states = npu_config.run_attention(query, key, value, attention_mask, "BSH",
                                                             head_dim, attn.heads)

                    hidden_states = npu_config.restore_dtype(hidden_states)

                hidden_states = hidden_states.reshape(hidden_states.shape[0], -1, attn.heads, head_dim)
                if self.sparse1d:
                    hidden_states = self._reverse_sparse_1d_on_npu(hidden_states, frame, height, width, pad_len)
                elif self.sparse2d:
                    hidden_states = self._reverse_sparse_2d(hidden_states, frame, height, width, pad_height, pad_width)

                hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if get_sequence_parallel_state():
                query = query.reshape(-1, attn.heads, head_dim)  # [s // sp, b, h * d] -> [s // sp * b, h, d]
                key = key.reshape(-1, attn.heads, head_dim)
                value = value.reshape(-1, attn.heads, head_dim)
                
                h_size = attn.heads * head_dim
                sp_size = nccl_info.world_size
                h_size_sp = h_size // sp_size
                # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
                query = all_to_all_SBH(query, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, h_size_sp)
                key = all_to_all_SBH(key, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, h_size_sp)
                value = all_to_all_SBH(value, scatter_dim=1, gather_dim=0).reshape(-1, batch_size, h_size_sp)
                query = query.reshape(-1, batch_size, attn.heads // sp_size, head_dim)
                key = key.reshape(-1, batch_size, attn.heads // sp_size, head_dim)
                value = value.reshape(-1, batch_size, attn.heads // sp_size, head_dim)

                if not self.is_cross_attn:
                    # require the shape of (batch_size x nheads x ntokens x dim)
                    pos_thw = self.position_getter(batch_size, t=frame * sp_size, h=height, w=width, device=query.device)
                    query = self.rope(query, pos_thw)
                    key = self.rope(key, pos_thw)

                query = rearrange(query, 's b h d -> b h s d')
                key = rearrange(key, 's b h d -> b h s d')
                value = rearrange(value, 's b h d -> b h s d')


                if self.sparse1d:
                    query, attention_mask, pad_len = self._sparse_1d(query, attention_mask, frame * sp_size, height, width)
                    
                    if self.is_cross_attn:
                        key = self._sparse_1d_kv(key)
                        value = self._sparse_1d_kv(value)
                    else:
                        key, _, pad_len = self._sparse_1d(key, None, frame * sp_size, height, width)
                        value, _, pad_len = self._sparse_1d(value, None, frame * sp_size, height, width)

                elif self.sparse2d:
                    query, attention_mask, pad_height, pad_width = self._sparse_2d(query, attention_mask, frame * sp_size, height, width)
                    if self.is_cross_attn:
                        key = self._sparse_2d_kv(key)
                        value = self._sparse_2d_kv(value)
                    else:
                        key, _, pad_height, pad_width = self._sparse_2d(key, None, frame * sp_size, height, width)
                        value, _, pad_height, pad_width = self._sparse_2d(value, None, frame * sp_size, height, width)
                

                # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
                # 0, 0 ->(bool) False, False ->(any) False ->(not) True
                if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
                    attention_mask = None
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
                if self.sparse1d:
                    hidden_states = self._reverse_sparse_1d(hidden_states, frame * sp_size, height, width, pad_len)
                elif self.sparse2d:
                    hidden_states = self._reverse_sparse_2d(hidden_states, frame * sp_size, height, width, pad_height, pad_width)
                
                hidden_states = rearrange(hidden_states, 'b h s d -> s b h d')

                hidden_states = hidden_states.reshape(-1, attn.heads // sp_size, head_dim)
                hidden_states = hidden_states.contiguous()
                # [s * b, h // sp, d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
                hidden_states = all_to_all_SBH(hidden_states, scatter_dim=0, gather_dim=1).reshape(-1, batch_size, h_size)
            else:
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                # qk norm
                # query = attn.q_norm(query)
                # key = attn.k_norm(key)

                if not self.is_cross_attn:
                    # require the shape of (batch_size x nheads x ntokens x dim)
                    pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
                    query = self.rope(query, pos_thw)
                    key = self.rope(key, pos_thw)
                
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                if self.sparse1d:
                    query, attention_mask, pad_len = self._sparse_1d(query, attention_mask, frame, height, width)
                    
                    if self.is_cross_attn:
                        key = self._sparse_1d_kv(key)
                        value = self._sparse_1d_kv(value)
                    else:
                        key, _, pad_len = self._sparse_1d(key, None, frame, height, width)
                        value, _, pad_len = self._sparse_1d(value, None, frame, height, width)

                elif self.sparse2d:
                    query, attention_mask, pad_height, pad_width = self._sparse_2d(query, attention_mask, frame, height, width)
                    if self.is_cross_attn:
                        key = self._sparse_2d_kv(key)
                        value = self._sparse_2d_kv(value)
                    else:
                        key, _, pad_height, pad_width = self._sparse_2d(key, None, frame, height, width)
                        value, _, pad_height, pad_width = self._sparse_2d(value, None, frame, height, width)
                # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
                # 0, 0 ->(bool) False, False ->(any) False ->(not) True
                if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
                    attention_mask = None
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
                if self.sparse1d:
                    hidden_states = self._reverse_sparse_1d(hidden_states, frame, height, width, pad_len)
                elif self.sparse2d:
                    hidden_states = self._reverse_sparse_2d(hidden_states, frame, height, width, pad_height, pad_width)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
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
        sparse2d: bool = False,
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
            sparse2d=sparse2d,
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
            sparse2d=sparse2d,
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
        batch_size = hidden_states.shape[0]

        if get_sequence_parallel_state():
            batch_size = hidden_states.shape[1]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
            ).chunk(6, dim=0)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)

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
