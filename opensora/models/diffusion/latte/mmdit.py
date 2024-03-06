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
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

import torch.utils.checkpoint as cp

# for i in sys.path:
#     print(i)

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

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
    
# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class JointAttention(nn.Module):
    def __init__(self, txt_dim, pix_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math', eps=1e-12, causal=True, ring_bucket_size=1024):
        super().__init__()
        dim = txt_dim + pix_dim
        assert txt_dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert pix_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv_text = nn.Linear(txt_dim, txt_dim * 3, bias=qkv_bias)
        self.qkv_pix = nn.Linear(pix_dim, pix_dim * 3, bias=qkv_bias)
        self.rms_q_text = RMSNorm(txt_dim)
        self.rms_k_text = RMSNorm(txt_dim)
        self.rms_q_pix = RMSNorm(pix_dim)
        self.rms_k_pix = RMSNorm(pix_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_pix = nn.Linear(dim, txt_dim)
        self.proj_text = nn.Linear(dim, pix_dim)
        self.proj_drop_text = nn.Dropout(proj_drop)
        self.proj_drop_pix = nn.Dropout(proj_drop)
        self.eps = eps
        self.causal = causal
        self.ring_bucket_size = ring_bucket_size

    def forward(self, x, c):
        B, N, C1 = x.shape
        qkv_pix = self.qkv_pix(x).reshape(B, N, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q_pix, k_pix, v_pix = qkv_pix.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_pix = self.rms_q_pix(q_pix)
        k_pix = self.rms_k_pix(k_pix)

        # Assuming 
        B, N, C2 = c.shape
        qkv_text = self.qkv_text(x).reshape(B, N, 3, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q_text, k_text, v_text = qkv_text.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if self.attention_mode != 'rebased':
            # Rebased does RMS norm inside already
            q_text = self.rms_q_text(q_text)
            k_text = self.rms_k_text(k_text)

        q = torch.cat([q_text, q_pix], dim=-1)
        k = torch.cat([k_text, k_pix], dim=-1)
        v = torch.cat([v_text, v_pix], dim=-1)

        C = C1 + C2
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            z = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                z = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p, scale=self.scale).reshape(B, N, C) # require pytorch 2.0
                
        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            z = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.attention_mode == 'rebased':
            z = parallel_rebased(q, k, v, self.eps, True, True).reshape(B, N, C)

        elif self.attention_mode == 'ring':
            z = ring_flash_attn_cuda(q, k, v, causal=self.causal, bucket_size=self.ring_bucket_size).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj_pix(z)
        x = self.proj_drop_pix(x)

        c = self.proj_text(z)
        c = self.proj_drop_text(c)

        return x, c


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class TextEmbeddingProjector(nn.Module): # Using instead of Label Embedder
    def __init__(self, clip_in_text_channels, t5_in_text_channels, text_hidden_size, timestep_hidden_size, bias=True):
        self.proj = nn.Linear(clip_in_text_channels + t5_in_text_channels, text_hidden_size, bias=bias)
        self.timestep_proj = nn.Linear(clip_in_text_channels, timestep_hidden_size, bias=bias)
    
    def forward(self,
                clip_embeds, # may be concat of multiple clip outputs (change the proj too then)
                t5_embeds):
        # Batch Len Chans
        B, L, C = clip_embeds.shape
        _, _, C_T5 = t5_embeds.shape

        pooled_clip_embeds = clip_embeds.permute(1, 0, 2)[0] # SDXL takes 0th item as the pooled version
        ts_proj = self.timestep_proj(pooled_clip_embeds)

        clip_embeds = nn.functional.pad(clip_embeds, (0, 0, C_T5 - C), mode='constant', value=0)
        embeds = torch.cat([clip_embeds, t5_embeds], dim=1) # concat in L dim
        c = self.proj(embeds)

        return c, ts_proj

#################################################################################
#                                 Core MMDiT/Latte Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A MMDiT/Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, y_size, text_hidden_size, pix_hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.num_params = 6
        self.text_y_input = nn.Sequential(
            nn.SiLU(),
            nn.Linear(y_size, self.num_params, bias=False)
        )
        self.pix_y_input = nn.Sequential(
            nn.SiLU(),
            nn.Linear(y_size, self.num_params, bias=False)
        )

        self.text_norm1 = nn.LayerNorm(text_hidden_size, elementwise_affine=False, eps=1e-6)
        self.pix_norm1 = nn.LayerNorm(pix_hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = JointAttention(text_hidden_size, pix_hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.text_norm2 = nn.LayerNorm(text_hidden_size, elementwise_affine=False, eps=1e-6)
        self.pix_norm2 = nn.LayerNorm(pix_hidden_size, elementwise_affine=False, eps=1e-6)

        text_mlp_hidden_dim = int(text_hidden_size * mlp_ratio)
        pix_mlp_hidden_dim = int(pix_hidden_size * mlp_ratio)

        approx_gelu = lambda: nn.Identity() #nn.GELU(approximate="tanh")
        self.text_mlp = Mlp(in_features=text_hidden_size, hidden_features=text_mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.pix_mlp = Mlp(in_features=pix_hidden_size, hidden_features=pix_mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, y, x, c):

        def bc(tnsr):
            return tnsr.view(-1, 1, 1, 1, 1) # broadcasting

        coefs_text = self.text_y_input(y)
        coefs_pix = self.pix_y_input(y)

        alpha_txt, beta_txt, gamma_txt, delta_txt, eps_txt, dz_txt = coefs_text.unbind(-1)
        alpha_pix, beta_pix, gamma_pix, delta_pix, eps_pix, dz_pix = coefs_pix.unbind(-1)

        c = self.text_norm1(c)
        x = self.pix_norm1(x)

        c = bc(alpha_txt) * c + beta_txt * torch.ones_like(alpha_txt)
        x = bc(alpha_pix) * x + beta_pix * torch.ones_like(alpha_pix)

        x, c = self.attn(x, c)
        x_new = self.pix_norm2(x * bc(gamma_pix) + x) * bc(delta_pix) + eps_pix * torch.ones_like(x)
        c_new = self.text_norm2(c * bc(gamma_txt) + c) * bc(delta_txt) + eps_txt * torch.ones_like(c)

        x_new = self.pix_mlp(x_new) * bc(dz_pix) + x
        c_new = self.text_mlp(c_new) * bc(dz_txt) + c

        return x, c


class FinalLayer(nn.Module):
    """
    The final layer of MMDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        clip_in_text_channels=384, # FIXME check the actual amount (note: it is the total number (i.e. after concating clips))
        clip_seq_len=77,
        t5_in_text_channels=4096,
        t5_seq_len=77,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        learn_sigma=True,
        extras=1,
        attention_mode='rebased',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames
        self.gradient_checkpointing = False

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, flatten=True)
        self.c_embedder = TextEmbeddingProjector(clip_seq_len*clip_in_text_channels, t5_seq_len*t5_in_text_channels, hidden_size, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            # timestep_size, text_hidden size, pix_hidden size
            TransformerBlock(hidden_size, hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # TODO (see latte.py)
        ...

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                t, 
                y=None, 
                text_embedding=None, 
                use_fp16=False):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)

        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(t, use_fp16=use_fp16)

        c, ts_proj = self.c_embedder(text_embedding.reshape(batches, -1))
        text_embedding_spatial = repeat(c, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        text_embedding_temp = repeat(c, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        t = t + ts_proj
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1]) 
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]

            c = timestep_spatial + text_embedding_spatial

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(spatial_block), x, c)
            else:
                x = spatial_block(x, c)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed

            c = timestep_temp + text_embedding_temp

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(temp_block), x, c)
            else:
                x = temp_block(x, c)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

        c = timestep_spatial
        x = self.final_layer(x, c)               
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False, text_embedding=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y=y, use_fp16=use_fp16, text_embedding=text_embedding)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   Latte Configs                                  #
#################################################################################

def Latte_XL_2(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def Latte_XL_4(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def Latte_XL_8(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def Latte_L_2(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Latte_L_4(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def Latte_L_8(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def Latte_B_2(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Latte_B_4(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def Latte_B_8(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def Latte_S_2(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def Latte_S_4(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def Latte_S_8(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    'Latte-XL/2': Latte_XL_2,  'Latte-XL/4': Latte_XL_4,  'Latte-XL/8': Latte_XL_8,
    'Latte-L/2':  Latte_L_2,   'Latte-L/4':  Latte_L_4,   'Latte-L/8':  Latte_L_8,
    'Latte-B/2':  Latte_B_2,   'Latte-B/4':  Latte_B_4,   'Latte-B/8':  Latte_B_8,
    'Latte-S/2':  Latte_S_2,   'Latte-S/4':  Latte_S_4,   'Latte-S/8':  Latte_S_8,
}

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    network = Latte_XL_2().to(device)
    from thop import profile 
    flops, params = profile(network, inputs=(img, t))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    # out = network(img, t, y)
    # print(out.shape)
