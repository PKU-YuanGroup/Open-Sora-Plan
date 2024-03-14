# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Union
from timm.layers import to_2tuple
from timm.models.vision_transformer import Mlp

from .pos import VisionRotaryEmbeddingFast

try:
    # needs to have https://github.com/corl-team/rebased/ installed
    from fla.ops.triton.rebased_fast import parallel_rebased
except:
    REBASED_IS_AVAILABLE = False

try:
    # needs to have https://github.com/lucidrains/ring-attention-pytorch installed
    from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda
except:
    RING_ATTENTION_IS_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################
#
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_lora=False,
                 attention_mode='math',
                 eps=1e-12,
                 causal=True,
                 ring_bucket_size=1024,
                 attention_pe_mode=None,
                 hw: Union[int, Tuple[int, int]] = 16,  # (h, w)
                 pt_hw: Union[int, Tuple[int, int]] = 16,  # (h, w)
                 intp_vfreq: bool = True,  # vision position interpolation
                 compress_kv: bool = False
                 ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.attention_pe_mode = attention_pe_mode

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.eps = eps
        self.causal = causal
        self.ring_bucket_size = ring_bucket_size

        if self.attention_pe_mode == '2d_rope':
            half_head_dim = dim // num_heads // 2
            self.hw = to_2tuple(hw)
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_hw=to_2tuple(pt_hw),
                ft_hw=self.hw if intp_vfreq else None,
            )

    def forward(self, x, attn_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) b h n c

        if self.attention_pe_mode == '2d_rope':
            q_t = q.view(B, self.num_heads, -1, self.hw[0] * self.hw[1], C // self.num_heads)
            ro_q_t = self.rope(q_t)
            q = ro_q_t.view(B, self.num_heads, N, C // self.num_heads)

            k_t = k.view(B, self.num_heads, -1, self.hw[0] * self.hw[1], C // self.num_heads)
            ro_k_t = self.rope(k_t)
            k = ro_k_t.view(B, self.num_heads, N, C // self.num_heads)

        if self.attention_mode == 'xformers':  # require pytorch 2.0
            if attn_mask is not None:
                attn_mask = self.make_attn_mask(attn_mask)
                attn_mask = attn_mask.repeat(1, self.num_heads, 1, 1).to(q.dtype)
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                   dropout_p=self.attn_drop.p, scale=self.scale).reshape(B, N, C)

        elif self.attention_mode == 'flash':  # require pytorch 2.0
            # https://github.com/PKU-YuanGroup/Open-Sora-Plan/issues/109
            assert attn_mask is None or torch.all(attn_mask.bool()), 'flash-attention do not support attention_mask'
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p, scale=self.scale).reshape(B, N, C)

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_bias = self.make_attn_bias(attn_mask)
                attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1).to(q.dtype)
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            if torch.any(torch.isnan(attn)):
                print('torch.any(torch.isnan(attn))')
                attn = attn.masked_fill(torch.isnan(attn), float(0.))
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.attention_mode == 'rebased':
            x = parallel_rebased(q, k, v, self.eps, True, True).reshape(B, N, C)

        elif self.attention_mode == 'ring':
            x = ring_flash_attn_cuda(q, k, v, causal=self.causal, bucket_size=self.ring_bucket_size).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def make_attn_mask(self, attn_mask):
        attn_mask = attn_mask.flatten(1).unsqueeze(-1)  # b n -> b n 1
        attn_mask = attn_mask @ attn_mask.transpose(1, 2)  # b n 1 @ b 1 n = b n n
        attn_mask = attn_mask.unsqueeze(1)  # b n n -> b 1 n n
        return attn_mask

    def make_attn_bias(self, attn_mask):
        # The numerical range of bfloat16, float16 can't conver -1e8
        # Refer to https://discuss.pytorch.org/t/runtimeerror-value-cannot-be-converted-to-type-at-half-without-overflow-1e-30/109768
        attn_mask = self.make_attn_mask(attn_mask)
        attn_bias = torch.where(attn_mask == 0, -1e8 if attn_mask.dtype == torch.float32 else -1e4, attn_mask)
        attn_bias = torch.where(attn_mask == 1, 0., attn_bias)
        return attn_bias

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0., attention_mode='math', **block_kwargs):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.d_model = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mode = attention_mode
        self.attn_drop = nn.Dropout(attn_drop)

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, attn_mask=None, cond_mask=None):
        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        kv = self.kv_linear(cond).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) b h n c

        if self.attention_mode == 'xformers' or self.attention_mode == 'flash':  # require pytorch 2.0
            attn_mask = self.make_attn_mask(attn_mask, cond_mask)
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(1, self.num_heads, 1, 1).to(q.dtype)
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                   dropout_p=self.attn_drop.p, scale=self.scale).reshape(B, N, C)

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn_bias = self.make_attn_bias(attn_mask, cond_mask)
            if attn_bias is not None:
                attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1).to(q.dtype)
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            if torch.any(torch.isnan(attn)):
                print('torch.any(torch.isnan(attn))')
                attn = attn.masked_fill(torch.isnan(attn), float(0.))
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def make_attn_mask(self, attn_mask, cond_mask):
        if attn_mask is None and cond_mask is None:
            return None
        if attn_mask is None and cond_mask is not None:
            attn_mask = cond_mask
        if attn_mask is not None and cond_mask is None:
            raise NotImplementedError
        attn_mask = attn_mask.flatten(1).unsqueeze(-1)  # b n -> b n 1
        cond_mask = cond_mask.flatten(1).unsqueeze(-1)  # b m -> b m 1
        attn_mask = attn_mask @ cond_mask.transpose(1, 2)  # b n 1 @ b 1 m = b n m
        attn_mask = attn_mask.unsqueeze(1)  # b n n -> b 1 n n
        return attn_mask

    def make_attn_bias(self, attn_mask, cond_mask):
        # The numerical range of bfloat16, float16 can't conver -1e8
        # Refer to https://discuss.pytorch.org/t/runtimeerror-value-cannot-be-converted-to-type-at-half-without-overflow-1e-30/109768
        attn_mask = self.make_attn_mask(attn_mask, cond_mask)
        if attn_mask is None:
            return None
        attn_bias = torch.where(attn_mask == 0, -1e8 if attn_mask.dtype == torch.float32 else -1e4, attn_mask)
        attn_bias = torch.where(attn_mask == 1, 0., attn_bias)
        return attn_bias

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Latte Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, extras=1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of Latte.
    """
    def __init__(self, hidden_size, patch_size_t, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size_t * patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
