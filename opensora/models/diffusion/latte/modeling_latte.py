from typing import Union, Tuple

import numpy as np
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange, repeat
from timm.layers import PatchEmbed, to_2tuple
from torch import nn
import torch
from opensora.models.diffusion.latte.pos import get_1d_sincos_temp_embed, get_2d_sincos_pos_embed
from opensora.models.diffusion.latte.common import TimestepEmbedder, LabelEmbedder, TransformerBlock, FinalLayer, \
    TransformerCrossConditonBlock, CaptionEmbedder

class Latte(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            input_size=(32, 32),
            patch_size=(2, 2),
            patch_size_t=1,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_frames=16,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
            extras=1,
            attention_mode='math',
            attention_pe_mode=None,
            pt_input_size: Union[int, Tuple[int, int]] = None,  # (h, w)
            pt_num_frames: Union[int, Tuple[int, int]] = None,  # (num_frames, 1)
            intp_vfreq: bool = True,  # vision position interpolation
            compress_kv: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = to_2tuple(patch_size)
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.extras = extras
        self.attention_mode = attention_mode
        self.attention_pe_mode = attention_pe_mode
        self.pt_input_size = pt_input_size
        self.pt_num_frames = pt_num_frames
        self.intp_vfreq = intp_vfreq
        self.compress_kv = compress_kv



        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.gradient_checkpointing = False

        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, self.hidden_size), requires_grad=False)

        if self.pt_input_size is None:
            self.pt_input_size = self.input_size
        if self.pt_num_frames is None:
            self.pt_num_frames = self.num_frames
        self.blocks = []
        for i in range(depth):
            if i % 2 == 0:
                m = TransformerBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, attention_mode=self.attention_mode,
                    attention_pe_mode=self.attention_pe_mode,
                    hw=(self.input_size[0] // self.patch_size[0], self.input_size[1] // self.patch_size[1]),
                    pt_hw=(self.pt_input_size[0] // self.patch_size[0], self.pt_input_size[1] // self.patch_size[1]),
                    intp_vfreq=self.intp_vfreq, compress_kv=self.compress_kv
                )
            else:
                m = TransformerBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, attention_mode=self.attention_mode,
                    attention_pe_mode=self.attention_pe_mode,
                    hw=(self.num_frames, 1),
                    pt_hw=(self.pt_num_frames, 1),
                    intp_vfreq=self.intp_vfreq, compress_kv=self.compress_kv
                )
            self.blocks.append(m)
        self.blocks = nn.ModuleList(self.blocks)

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size_t, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Latte blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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
                attn_mask=None):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        attn_mask: (N, F, H, W)
        """
        attn_mask_temproal, attn_mask_spatial = None, None
        if attn_mask is not None:
            attn_mask_spatial = rearrange(attn_mask, 'b t h w -> (b t) (h w)')
            attn_mask_temproal = rearrange(attn_mask, 'b t h w -> (b h w) t')

        batches, frames, channels, high, weight = x.shape

        x = rearrange(x, 'b f c h w -> (b f) c h w').to(self.pos_embed.dtype)

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t, self.pos_embed.dtype)

        # timestep condition
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])

        # class condition
        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=self.temp_embed.shape[1])
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        # uncondition
        else:
            pass

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            else:
                c = timestep_spatial


            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(spatial_block), x, c, attn_mask_spatial)
            else:
                x = spatial_block(x, c, attn_mask_spatial)


            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed

            if self.extras == 2:
                c = timestep_temp + y_temp
            else:
                c = timestep_temp

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(temp_block), x, c, attn_mask_temproal)
            else:
                x = temp_block(x, c, attn_mask_temproal)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

        if self.extras == 2:
            c = timestep_spatial + y_spatial
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)               
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)

        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, attn_mask=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y=y, attn_mask=attn_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)

class LatteT2V(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            input_size=(32, 32),
            patch_size=(2, 2),
            patch_size_t=1,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            num_frames=16,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
            extras=1,
            attention_mode='math',
            attention_pe_mode=None,
            pt_input_size: Union[int, Tuple[int, int]] = None,  # (h, w)
            pt_num_frames: Union[int, Tuple[int, int]] = None,  # (num_frames, 1)
            intp_vfreq: bool = True,  # vision position interpolation
            compress_kv: bool = False,
            caption_channels=4096,
            model_max_length=1024,
    ):
        super(LatteT2V, self).__init__()
        self.input_size = input_size
        self.patch_size = to_2tuple(patch_size)
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.extras = extras
        self.attention_mode = attention_mode
        self.attention_pe_mode = attention_pe_mode
        self.pt_input_size = pt_input_size
        self.pt_num_frames = pt_num_frames
        self.intp_vfreq = intp_vfreq
        self.compress_kv = compress_kv

        self.caption_channels = caption_channels
        self.model_max_length = model_max_length



        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.gradient_checkpointing = False

        self.x_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)

        # if self.extras == 2:
        #     self.y_embedder = LabelEmbedder(self.num_classes, self.hidden_size, self.class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, self.hidden_size), requires_grad=False)

        if self.pt_input_size is None:
            self.pt_input_size = self.input_size
        if self.pt_num_frames is None:
            self.pt_num_frames = self.num_frames
        self.blocks = []
        for i in range(self.depth):
            if i % 2 == 0:
                m = TransformerCrossConditonBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, attention_mode=self.attention_mode,
                    attention_pe_mode=self.attention_pe_mode,
                    hw=(self.input_size[0] // self.patch_size[0], self.input_size[1] // self.patch_size[1]),
                    pt_hw=(self.pt_input_size[0] // self.patch_size[0], self.pt_input_size[1] // self.patch_size[1]),
                    intp_vfreq=self.intp_vfreq, compress_kv=self.compress_kv
                )
            else:
                m = TransformerBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio, attention_mode=self.attention_mode,
                    attention_pe_mode=self.attention_pe_mode,
                    hw=(self.num_frames, 1),
                    pt_hw=(self.pt_num_frames, 1),
                    intp_vfreq=self.intp_vfreq, compress_kv=self.compress_kv
                )
            self.blocks.append(m)
        self.blocks = nn.ModuleList(self.blocks)

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(in_channels=self.caption_channels, hidden_size=self.hidden_size,
                                          uncond_prob=self.class_dropout_prob, act_layer=approx_gelu,
                                          token_num=self.model_max_length)

        self.final_layer = FinalLayer(self.hidden_size, self.patch_size_t, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Latte blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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

    def forward(self,
                x,
                t,
                cond,
                attn_mask=None,
                cond_mask=None):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        cond: (N, 1, L, C_) tensor of text conditions
        attn_mask: (N, F, H, W)
        cond_mask: (N, L)
        """
        attn_mask_temproal, attn_mask_spatial = None, None
        if attn_mask is not None:
            attn_mask_spatial = rearrange(attn_mask, 'b t h w -> (b t) (h w)')
            attn_mask_temproal = rearrange(attn_mask, 'b t h w -> (b h w) t')

        batches, frames, channels, high, weight = x.shape

        x = rearrange(x, 'b f c h w -> (b f) c h w').to(self.pos_embed.dtype)

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t, self.pos_embed.dtype)
        cond = self.y_embedder(cond, self.training)  # (N, 1, L, D)

        # timestep condition
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        cond_spatial = repeat(cond, 'b 1 l d -> (b f) l d', f=self.temp_embed.shape[1]).contiguous()
        cond_mask_spatial = repeat(cond_mask, 'b l -> (b f) l', f=self.temp_embed.shape[1]).contiguous()

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i + 2]

            # cond only apply to spatial attention
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(spatial_block), x, cond_spatial, timestep_spatial, attn_mask_spatial, cond_mask_spatial)
            else:
                x = spatial_block(x, cond_spatial, timestep_spatial, attn_mask_spatial, cond_mask_spatial)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed

            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(temp_block), x, timestep_temp, attn_mask_temproal)
            else:
                x = temp_block(x, timestep_temp, attn_mask_temproal)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

        x = self.final_layer(x, timestep_spatial)
        x = self.unpatchify(x)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x

#################################################################################
#                                   Latte Configs                                  #
#################################################################################


def Latte_XL_122(**kwargs):
    return Latte(depth=56, hidden_size=1152, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def Latte_XL_144(**kwargs):
    return Latte(depth=56, hidden_size=1152, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def Latte_XL_188(**kwargs):
    return Latte(depth=56, hidden_size=1152, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def Latte_L_122(**kwargs):
    return Latte(depth=48, hidden_size=1024, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def Latte_L_144(**kwargs):
    return Latte(depth=48, hidden_size=1024, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def Latte_L_188(**kwargs):
    return Latte(depth=48, hidden_size=1024, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def Latte_B_122(**kwargs):
    return Latte(depth=24, hidden_size=768, patch_size_t=1, patch_size=2, num_heads=12, **kwargs)

def Latte_B_144(**kwargs):
    return Latte(depth=24, hidden_size=768, patch_size_t=1, patch_size=4, num_heads=12, **kwargs)

def Latte_B_188(**kwargs):
    return Latte(depth=24, hidden_size=768, patch_size_t=1, patch_size=8, num_heads=12, **kwargs)

def Latte_S_122(**kwargs):
    return Latte(depth=24, hidden_size=384, patch_size_t=1, patch_size=2, num_heads=6, **kwargs)

def Latte_S_144(**kwargs):
    return Latte(depth=24, hidden_size=384, patch_size_t=1, patch_size=4, num_heads=6, **kwargs)

def Latte_S_188(**kwargs):
    return Latte(depth=24, hidden_size=384, patch_size_t=1, patch_size=8, num_heads=6, **kwargs)






def LatteT2V_XL_122(**kwargs):
    return LatteT2V(depth=56, hidden_size=1152, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def LatteT2V_XL_144(**kwargs):
    return LatteT2V(depth=56, hidden_size=1152, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def LatteT2V_XL_188(**kwargs):
    return LatteT2V(depth=56, hidden_size=1152, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def LatteT2V_L_122(**kwargs):
    return LatteT2V(depth=48, hidden_size=1024, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def LatteT2V_L_144(**kwargs):
    return LatteT2V(depth=48, hidden_size=1024, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def LatteT2V_L_188(**kwargs):
    return LatteT2V(depth=48, hidden_size=1024, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def LatteT2V_B_122(**kwargs):
    return LatteT2V(depth=24, hidden_size=768, patch_size_t=1, patch_size=2, num_heads=12, **kwargs)

def LatteT2V_B_144(**kwargs):
    return LatteT2V(depth=24, hidden_size=768, patch_size_t=1, patch_size=4, num_heads=12, **kwargs)

def LatteT2V_B_188(**kwargs):
    return LatteT2V(depth=24, hidden_size=768, patch_size_t=1, patch_size=8, num_heads=12, **kwargs)

def LatteT2V_S_122(**kwargs):
    return LatteT2V(depth=24, hidden_size=384, patch_size_t=1, patch_size=2, num_heads=6, **kwargs)

def LatteT2V_S_144(**kwargs):
    return LatteT2V(depth=24, hidden_size=384, patch_size_t=1, patch_size=4, num_heads=6, **kwargs)

def LatteT2V_S_188(**kwargs):
    return LatteT2V(depth=24, hidden_size=384, patch_size_t=1, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    "Latte-XL/122": Latte_XL_122, "Latte-XL/144": Latte_XL_144, "Latte-XL/188": Latte_XL_188,
    "Latte-L/122": Latte_L_122, "Latte-L/144": Latte_L_144, "Latte-L/188": Latte_L_188,
    "Latte-B/122": Latte_B_122, "Latte-B/144": Latte_B_144, "Latte-B/188": Latte_B_188,
    "Latte-S/122": Latte_S_122, "Latte-S/144": Latte_S_144, "Latte-S/188": Latte_S_188,
}


LatteT2V_models = {
    "LatteT2V-XL/122": LatteT2V_XL_122, "LatteT2V-XL/144": LatteT2V_XL_144, "LatteT2V-XL/188": LatteT2V_XL_188,
    "LatteT2V-L/122": LatteT2V_L_122, "LatteT2V-L/144": LatteT2V_L_144, "LatteT2V-L/188": LatteT2V_L_188,
    "LatteT2V-B/122": LatteT2V_B_122, "LatteT2V-B/144": LatteT2V_B_144, "LatteT2V-B/188": LatteT2V_B_188,
    "LatteT2V-S/122": LatteT2V_S_122, "LatteT2V-S/144": LatteT2V_S_144, "LatteT2V-S/188": LatteT2V_S_188,
}

if __name__ == '__main__':

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """        
    x: (N, F, C, H, W) tensor of video inputs
    t: (N,) tensor of diffusion timesteps
    cond: (N, L, C_) tensor of text conditions
    attn_mask: (N, F, H, W)
    cond_mask: (N, L)
    """
    x = torch.randn(2, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2]).to(device)
    cond = torch.randn(2, 1, 120, 4096).to(device)
    attn_mask = torch.ones(2, 16, 32//2, 32//2).to(device)
    cond_mask = torch.ones(2, 120).to(device)
    model = LatteT2V_XL_122().to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    for n, p in model.named_parameters():
        print(f"Training Parameters: {n}, {p.shape}")
    with torch.no_grad():
        out = model(x, t, cond, attn_mask, cond_mask)
        print(out.shape)  # torch.Size([2, 16, 8, 32, 32])