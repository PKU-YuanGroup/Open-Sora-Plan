from einops import rearrange
from torch import nn
import torch
import numpy as np

from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from diffusers.utils.torch_utils import maybe_allow_in_graph
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
import diffusers
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import GatedSelfAttentionDense
from diffusers.models.attention_processor import Attention as Attention_
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from .rope import PositionGetter3D, RoPE3D
import re
try:
    import torch_npu
    from opensora.npu_config import npu_config, set_run_dtype
except:
    torch_npu = None
    npu_config = None
    set_run_dtype = None
logger = logging.get_logger(__name__)

def get_3d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, 
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)
    grid_t = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_h = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid_w = np.arange(grid_size[2], dtype=np.float32) / (grid_size[2] / base_size[2]) / interpolation_scale[2]
    grid = np.meshgrid(grid_w, grid_h, grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[2], grid_size[1], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (T*H*W, D/3)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (T*H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, 
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size[0]) / interpolation_scale[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size[1]) / interpolation_scale[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16, 
):
    """
    grid_size: int of the grid return: pos_embed: [grid_size, embed_dim] or
    [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # if isinstance(grid_size, int):
    #     grid_size = (grid_size, grid_size)

    grid = np.arange(grid_size, dtype=np.float32) / (grid_size / base_size) / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, 
                            self.bias.float() if self.bias is not None else None, self.eps).to(origin_dtype)


class FP32_SiLU(nn.SiLU):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(inputs.float(), inplace=self.inplace).to(inputs.dtype)


class FP32_GELU(nn.GELU):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(inputs.float(), approximate=self.approximate).to(inputs.dtype)



class OverlapPatchEmbed3D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1, 
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=False, 
    ):
        super().__init__()
        # assert patch_size_t == 1 and patch_size == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=3, padding=1, stride=(patch_size_t, patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t)
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)
        # self.temp_embed_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        # assert latent.shape[-3] == 1 and num_frames == 1
        
        num_frames = latent.shape[-3] // self.patch_size_t
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        # latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)

        if self.flatten:
            # latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
            latent = rearrange(latent, 'b c t h w -> (b t) (h w) c ')
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed


            if self.num_frames != num_frames:
                # import ipdb;ipdb.set_trace()
                # raise NotImplementedError
                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)
        
        latent = rearrange(latent, '(b t) n c -> b t n c', b=b)
        video_latent = latent

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            video_latent = (video_latent + temp_pos_embed).to(video_latent.dtype)

        video_latent = rearrange(video_latent, 'b t n c -> b (t n) c')
        return video_latent
    


class OverlapPatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with 3D position embedding"""

    def __init__(
        self,
        num_frames=1, 
        height=224,
        width=224,
        patch_size_t=1,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=(1, 1),
        interpolation_scale_t=1,
        use_abs_pos=False, 
    ):
        super().__init__()
        assert patch_size_t == 1
        self.use_abs_pos = use_abs_pos
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, padding=1, stride=(patch_size, patch_size), bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161

        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = (height // patch_size, width // patch_size)
        self.interpolation_scale = (interpolation_scale[0], interpolation_scale[1])
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

        self.num_frames = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.base_size_t = (num_frames - 1) // patch_size_t + 1 if num_frames % 2 == 1 else num_frames // patch_size_t
        self.interpolation_scale_t = interpolation_scale_t
        temp_pos_embed = get_1d_sincos_pos_embed(embed_dim, self.num_frames, base_size=self.base_size_t, interpolation_scale=self.interpolation_scale_t)
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)
        # self.temp_embed_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, latent, num_frames):
        b, _, _, _, _ = latent.shape
        video_latent, image_latent = None, None
        # b c 1 h w
        # assert latent.shape[-3] == 1 and num_frames == 1
        num_frames = latent.shape[-3] // self.patch_size_t
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)

        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BT C H W -> BT N C
        if self.layer_norm:
            latent = self.norm(latent)

        if self.use_abs_pos:
            # Interpolate positional embeddings if needed.
            # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
            if self.height != height or self.width != width:
                # raise NotImplementedError
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed


            if self.num_frames != num_frames:
                # import ipdb;ipdb.set_trace()
                # raise NotImplementedError
                temp_pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.temp_pos_embed.shape[-1],
                    grid_size=num_frames,
                    base_size=self.base_size_t,
                    interpolation_scale=self.interpolation_scale_t,
                )
                temp_pos_embed = torch.from_numpy(temp_pos_embed)
                temp_pos_embed = temp_pos_embed.float().unsqueeze(0).to(latent.device)
            else:
                temp_pos_embed = self.temp_pos_embed

            latent = (latent + pos_embed).to(latent.dtype)
        
        latent = rearrange(latent, '(b t) n c -> b t n c', b=b)
        assert num_frames == latent.shape[1]

        if self.use_abs_pos:
            # temp_pos_embed = temp_pos_embed.unsqueeze(2) * self.temp_embed_gate.tanh()
            temp_pos_embed = temp_pos_embed.unsqueeze(2)
            latent = (latent + temp_pos_embed).to(latent.dtype)

        latent = rearrange(latent, 'b t n c -> b (t n) c')
        return latent
    

class Attention(Attention_):
    def __init__(self, downsampler, attention_mode, use_rope, interpolation_scale_thw, **kwags):
        processor = AttnProcessor2_0(attention_mode=attention_mode, use_rope=use_rope, interpolation_scale_thw=interpolation_scale_thw)
        super().__init__(processor=processor, **kwags)
        self.downsampler = None
        if downsampler: # downsampler  k155_s122
            downsampler_ker_size = list(re.search(r'k(\d{2,3})', downsampler).group(1)) # 122
            down_factor = list(re.search(r's(\d{2,3})', downsampler).group(1))
            downsampler_ker_size = [int(i) for i in downsampler_ker_size]
            downsampler_padding = [(i - 1) // 2 for i in downsampler_ker_size]
            down_factor = [int(i) for i in down_factor]
            
            if len(downsampler_ker_size) == 2:
                self.downsampler = DownSampler2d(kwags['query_dim'], kwags['query_dim'], kernel_size=downsampler_ker_size, stride=1,
                                            padding=downsampler_padding, groups=kwags['query_dim'], down_factor=down_factor,
                                            down_shortcut=True)
            elif len(downsampler_ker_size) == 3:
                self.downsampler = DownSampler3d(kwags['query_dim'], kwags['query_dim'], kernel_size=downsampler_ker_size, stride=1,
                                            padding=downsampler_padding, groups=kwags['query_dim'], down_factor=down_factor,
                                            down_shortcut=True)
        
        self.q_norm = nn.LayerNorm(kwags['dim_head'], elementwise_affine=True, eps=1e-6)
        self.k_norm = nn.LayerNorm(kwags['dim_head'], elementwise_affine=True, eps=1e-6) 
        

class DownSampler3d(nn.Module):
    def __init__(self, *args, **kwargs):
        ''' Required kwargs: down_factor, downsampler'''
        super().__init__()
        self.down_factor = kwargs.pop('down_factor')
        self.down_shortcut = kwargs.pop('down_shortcut')
        self.layer = nn.Conv3d(*args, **kwargs)

    def forward(self, x, attention_mask, t, h, w):
        b = x.shape[0]
        x = rearrange(x, 'b (t h w) d -> b d t h w', t=t, h=h, w=w)
        if npu_config is None:
            x = self.layer(x) + (x if self.down_shortcut else 0)
        else:
            x_dtype = x.dtype
            x = npu_config.run_conv3d(self.layer, x, x_dtype) + (x if self.down_shortcut else 0)
        
        self.t = t//self.down_factor[0]
        self.h = h//self.down_factor[1]
        self.w = w//self.down_factor[2]
        # import ipdb;ipdb.set_trace()
        x = rearrange(x, 'b d (t dt) (h dh) (w dw) -> (b dt dh dw) (t h w) d', 
                      t=t//self.down_factor[0], h=h//self.down_factor[1], w=w//self.down_factor[2], 
                         dt=self.down_factor[0], dh=self.down_factor[1], dw=self.down_factor[2])
    

        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> b 1 t h w', t=t, h=h, w=w)
        attention_mask = rearrange(attention_mask, 'b 1 (t dt) (h dh) (w dw) -> (b dt dh dw) 1 (t h w)',
                      t=t//self.down_factor[0], h=h//self.down_factor[1], w=w//self.down_factor[2], 
                         dt=self.down_factor[0], dh=self.down_factor[1], dw=self.down_factor[2])
        return x, attention_mask
    
    def reverse(self, x, t, h, w):
        # import ipdb;ipdb.set_trace()
        x = rearrange(x, '(b dt dh dw) (t h w) d -> b (t dt h dh w dw) d', 
                      t=t, h=h, w=w, 
                         dt=self.down_factor[0], dh=self.down_factor[1], dw=self.down_factor[2])
        return x



class DownSampler2d(nn.Module):
    def __init__(self, *args, **kwargs):
        ''' Required kwargs: down_factor, downsampler'''
        super().__init__()
        self.down_factor = kwargs.pop('down_factor')
        self.down_shortcut = kwargs.pop('down_shortcut')
        self.layer = nn.Conv2d(*args, **kwargs)

    def forward(self, x, attention_mask, t, h, w):
        # import ipdb;ipdb.set_trace()
        b = x.shape[0]
        x = rearrange(x, 'b (t h w) d -> (b t) d h w', t=t, h=h, w=w)
        x = self.layer(x) + (x if self.down_shortcut else 0)

        self.t = 1
        self.h = h//self.down_factor[0]
        self.w = w//self.down_factor[1]

        x = rearrange(x, 'b d (h dh) (w dw) -> (b dh dw) (h w) d', 
                      h=h//self.down_factor[0], w=w//self.down_factor[1], 
                      dh=self.down_factor[0], dw=self.down_factor[1])
    
        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> (b t) 1 h w', h=h, w=w)
        attention_mask = rearrange(attention_mask, 'b 1 (h dh) (w dw) -> (b dh dw) 1 (h w)',
                      h=h//self.down_factor[0], w=w//self.down_factor[1], 
                         dh=self.down_factor[0], dw=self.down_factor[1])
        return x, attention_mask
    
    def reverse(self, x, t, h, w):
        x = rearrange(x, '(b t dh dw) (h w) d -> b (t h dh w dw) d', 
                      t=t, h=h, w=w, 
                      dh=self.down_factor[0], dw=self.down_factor[1])
        return x
    

class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attention_mode='xformers', use_rope=False, interpolation_scale_thw=(1, 1, 1)):
        self.use_rope = use_rope
        self.interpolation_scale_thw = interpolation_scale_thw
        if self.use_rope:
            self._init_rope(interpolation_scale_thw)
        self.attention_mode = attention_mode
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _init_rope(self, interpolation_scale_thw):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D()

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
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        if attn.downsampler is not None:
            hidden_states, attention_mask = attn.downsampler(hidden_states, attention_mask, t=frame, h=height, w=width)
            frame, height, width = attn.downsampler.t, attn.downsampler.h, attn.downsampler.w

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # if input_ndim == 4:
        #     batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            if npu_config is None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            else:
                if npu_config.enable_FA:
                    attention_mask = attention_mask.to(torch.bool)
                attention_mask = attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])
                attention_mask = attention_mask.repeat(1, 1, hidden_states.shape[1], 1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if npu_config is not None and npu_config.on_npu:
            if npu_config.enable_FA and query.dtype == torch.float32:
                dtype = torch.bfloat16
            else:
                dtype = None

            if self.use_rope:
                query = query.view(batch_size, -1, attn.heads, head_dim)
                key = key.view(batch_size, -1, attn.heads, head_dim)
                # require the shape of (batch_size x nheads x ntokens x dim)
                pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)
                query = query.view(batch_size, -1, attn.heads * head_dim)
                key = key.view(batch_size, -1, attn.heads * head_dim)

            with set_run_dtype(query, dtype):
                query, key, value = npu_config.set_current_run_dtype([query, key, value])
                hidden_states = npu_config.run_attention(query, key, value, attention_mask, "BSH",
                                                         head_dim, attn.heads)

                hidden_states = npu_config.restore_dtype(hidden_states)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # qk norm
            query = attn.q_norm(query)
            key = attn.k_norm(key)

            if self.use_rope:
                # require the shape of (batch_size x nheads x ntokens x dim)
                pos_thw = self.position_getter(batch_size, t=frame, h=height, w=width, device=query.device)
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)

            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
            # 0, 0 ->(bool) False, False ->(any) False ->(not) True
            if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
                attention_mask = None
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            
            if self.attention_mode == 'flash':
                assert attention_mask is None, 'flash-attn do not support attention_mask'
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
            elif self.attention_mode == 'xformers':
                with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
            elif self.attention_mode == 'math':
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            else:
                raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if attn.downsampler is not None:
            hidden_states = attn.downsampler.reverse(hidden_states, t=frame, h=height, w=width)

        return hidden_states


class PixelUnshuffle(nn.Module):
    def __init__(self, ratio, ratio_t=None):
        super().__init__()
        self.r = ratio
        self.r_t = ratio_t if ratio_t else 1

    def forward(self, x):
        # if self.r_t is not None and self.r_t != 1:
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            # import ipdb;ipdb.set_trace()
            assert t % self.r_t == 0 and h % self.r == 0 and w % self.r == 0
            x = rearrange(x, 'b c (t r1) (h r2) (w r3) -> b (c r1 r2 r3) t h w', r1=self.r_t, r2=self.r, r3=self.r)
        else:
            b, c, h, w = x.shape
            assert h % self.r == 0 and w % self.r == 0
            x = rearrange(x, 'b c (h r2) (w r3) -> b (c r2 r3) h w', r2=self.r, r3=self.r)
        return x
    
class PixelShuffle(nn.Module):
    def __init__(self, ratio, ratio_t=None):
        super().__init__()
        self.r = ratio
        self.r_t = ratio_t if ratio_t else 1

    def forward(self, x):
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            # import ipdb;ipdb.set_trace()
            assert c % (self.r_t*self.r*self.r) == 0
            x = rearrange(x, 'b (c r1 r2 r3) t h w -> b c (t r1) (h r2) (w r3)', r1=self.r_t, r2=self.r, r3=self.r)
        else:
            b, c, h, w = x.shape
            assert c % (self.r*self.r) == 0
            x = rearrange(x, 'b (c r2 r3) h w -> b c (h r2) (w r3)', r2=self.r, r3=self.r)
        return x

class Downsample3d(nn.Module):
    def __init__(self, n_feat):
        super(Downsample3d, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUnshuffle(2, 2))

    def forward(self, x, attention_mask, frames, height, width, pad_h=0, pad_w=0):
        x = rearrange(x, 'b (t h w) d -> b d t h w', t=frames, h=height, w=width)
        # x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode='reflect')
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0))
        if npu_config is None:
            x = self.body(x)
        else:
            x_dtype = x.dtype
            x = npu_config.run_conv3d(self.body, x, x_dtype)
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        
        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> b 1 t h w', t=frames, h=height, w=width)
        attention_mask = F.pad(attention_mask, (0, pad_w, 0, pad_h, 0, 0))
        attention_mask = F.max_pool3d(attention_mask, kernel_size=2, stride=2)
        attention_mask = rearrange(attention_mask, 'b 1 t h w -> b 1 (t h w)')
        attention_bias = (1 - attention_mask.bool().to(x.dtype)) * -10000.0

        return x, attention_bias, attention_mask
    
class Upsample3d(nn.Module):
    def __init__(self, n_feat):
        super(Upsample3d, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle(2, 2))

    def forward(self, x, attention_mask, frames, height, width, pad_h=0, pad_w=0):
        x = rearrange(x, 'b (t h w) d -> b d t h w', t=frames, h=height, w=width)
        if npu_config is None:
            x = self.body(x)
        else:
            x_dtype = x.dtype
            x = npu_config.run_conv3d(self.body, x, x_dtype)
        x = x[:, :, :, :height*2-pad_h, :width*2-pad_w]
        x = rearrange(x, 'b d t h w -> b (t h w) d')

        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> b 1 t h w', t=frames, h=height, w=width)
        attention_mask = attention_mask.repeat_interleave(2, -1).repeat_interleave(2, -2).repeat_interleave(2, -3)
        attention_mask = attention_mask[:, :, :, :height*2-pad_h, :width*2-pad_w]
        attention_mask = rearrange(attention_mask, 'b 1 t h w -> b 1 (t h w)')
        attention_bias = (1 - attention_mask.bool().to(x.dtype)) * -10000.0

        return x, attention_bias, attention_mask


class Downsample2d(nn.Module):
    def __init__(self, n_feat, is_video_model=False):
        super(Downsample2d, self).__init__()
        self.is_video_model = is_video_model
        Conv = nn.Conv3d if is_video_model else nn.Conv2d
        self.body = nn.Sequential(Conv(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUnshuffle(2, 1))

    def forward(self, x, attention_mask, frames, height, width, pad_h=0, pad_w=0):
        # import ipdb;ipdb.set_trace()
        x = rearrange(x, 'b (t h w) d -> (b t) d h w', t=frames, h=height, w=width)
        # x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        # x = F.pad(x, (0, pad_w, 0, pad_h))
        if npu_config is None:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x = npu_config.run_pad_2d(F.pad, x, pad=(0, pad_w, 0, pad_h))
        if self.is_video_model:
            x = rearrange(x, '(b t) d h w -> b d t h w', t=frames)
        x = self.body(x)
        if self.is_video_model:
            x = rearrange(x, 'b d t h w -> b (t h w) d', t=frames)
        else:
            x = rearrange(x, '(b t) d h w -> b (t h w) d', t=frames)
        
        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> (b t) 1 h w', t=frames, h=height, w=width)
        attention_mask = F.pad(attention_mask, (0, pad_w, 0, pad_h))
        attention_mask = F.max_pool2d(attention_mask.float(), kernel_size=2, stride=2)
        attention_mask = rearrange(attention_mask, '(b t) 1 h w -> b 1 (t h w)', t=frames)
        attention_bias = (1 - attention_mask.bool().to(x.dtype)) * -10000.0

        return x, attention_bias, attention_mask
    
class Upsample2d(nn.Module):
    def __init__(self, n_feat, is_video_model=False):
        super(Upsample2d, self).__init__()
        self.is_video_model = is_video_model
        Conv = nn.Conv3d if is_video_model else nn.Conv2d
        self.body = nn.Sequential(Conv(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle(2, 1))

    def forward(self, x, attention_mask, frames, height, width, pad_h=0, pad_w=0):
        if self.is_video_model:
            x = rearrange(x, 'b (t h w) d -> b d t h w', t=frames, h=height, w=width)
        else:
            x = rearrange(x, 'b (t h w) d -> (b t) d h w', t=frames, h=height, w=width)
        x = self.body(x)
        if self.is_video_model:
            x = rearrange(x, 'b d t h w -> (b t) d h w')
        x = x[:, :, :height*2-pad_h, :width*2-pad_w]
        x = rearrange(x, '(b t) d h w -> b (t h w) d', t=frames)

        attention_mask = rearrange(attention_mask, 'b 1 (t h w) -> (b t) 1 h w', t=frames, h=height, w=width)
        attention_mask = attention_mask.repeat_interleave(2, -1).repeat_interleave(2, -2)
        attention_mask = attention_mask[:, :, :height*2-pad_h, :width*2-pad_w]
        attention_mask = rearrange(attention_mask, '(b t) 1 h w -> b 1 (t h w)', t=frames)
        attention_bias = (1 - attention_mask.bool().to(x.dtype)) * -10000.0

        return x, attention_bias, attention_mask


class FeedForward_Conv2d(nn.Module):
    def __init__(self, downsampler, dim, hidden_features, bias=True):
        super(FeedForward_Conv2d, self).__init__()
        
        self.bias = bias

        self.project_in = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(5, 5), stride=1, padding=(2, 2), dilation=1,
                        groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=1,
                        groups=hidden_features, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 1), stride=1, padding=(0, 0), dilation=1,
                        groups=hidden_features, bias=bias)
        ])

        self.project_out = nn.Linear(hidden_features, dim, bias=bias)


    def forward(self, x, t, h, w):
        # import ipdb;ipdb.set_trace()
        x = self.project_in(x)
        x = rearrange(x, 'b (t h w) d -> (b t) d h w', t=t, h=h, w=w)
        
        x = self.act(x)
        out = x
        for module in self.dwconv:
            out = out + module(x)
        out = rearrange(out, '(b t) d h w -> b (t h w) d', t=t, h=h, w=w)
        x = self.project_out(out)
        return x

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        attention_mode: str = "xformers", 
        downsampler: str = None, 
        mlp_ratio: int = 4, 
        use_rope: bool = False,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1), 
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
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
            downsampler=downsampler, 
            attention_mode=attention_mode, 
            use_rope=use_rope, 
            interpolation_scale_thw=interpolation_scale_thw, 
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
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
                downsampler=None, 
                attention_mode=attention_mode, 
                use_rope=False, 
                interpolation_scale_thw=interpolation_scale_thw, 
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward_Conv2d(
            downsampler, 
            dim,
            hidden_features=mlp_ratio * dim, 
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
        
        # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
        # 0, 0 ->(bool) False, False ->(any) False ->(not) True
        # assert attention_mask.bool().float().sum() / attention_mask.bool().float().numel() <= 1/16, 'must ~all visible'
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask, frame=frame, height=height, width=width, 
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states, t=frame, h=height, w=width)
        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states
