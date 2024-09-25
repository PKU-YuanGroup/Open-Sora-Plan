import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention as Attention_
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



# try:
#     from .curope import cuRoPE3D
#     RoPE3D = cuRoPE3D


    
#     class PositionGetter3D(object):
#         """ return positions of patches """

#         def __init__(self):
#             self.cache_positions = {}
            
#         def __call__(self, b, t, h, w, device):
#             if not (t,h,w) in self.cache_positions:
#                 x = torch.arange(w, device=device)
#                 y = torch.arange(h, device=device)
#                 z = torch.arange(t, device=device)
#                 self.cache_positions[t,h,w] = torch.cartesian_prod(z, y, x) # (t, h, w, 3)
#             pos = self.cache_positions[t,h,w].view(1, t*h*w, 3).expand(b, -1, 3).clone()
#             return pos
        
# except ImportError:
    # print('Warning, cannot find cuda-compiled version of RoPE3D, using a slow pytorch version instead')


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
            # print('PositionGetter3D', PositionGetter3D)
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

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        # for (ntokens x batch_size x nheads x dim)
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: ntokens x batch_size x nheads x dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (ntokens x batch_size x nheads x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2# Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens