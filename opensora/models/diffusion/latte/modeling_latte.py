import numpy as np
from einops import rearrange, repeat
from timm.layers import PatchEmbed
from torch import nn
import torch
from .pos import get_1d_sincos_temp_embed, get_2d_sincos_pos_embed
from .common import TimestepEmbedder, LabelEmbedder, TransformerBlock, FinalLayer
from .configuration_latte import LatteConfiguration

class Latte(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(self, config: LatteConfiguration):
        super().__init__()

        input_size = config.input_size
        patch_size = config.patch_size
        patch_size_t = config.patch_size_t
        in_channels = config.in_channels
        hidden_size = config.hidden_size
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        num_frames = config.num_frames
        class_dropout_prob = config.class_dropout_prob
        num_classes = config.num_classes
        learn_sigma = config.learn_sigma
        extras = config.learn_sigma
        attention_mode = config.attention_mode
        compress_kv = config.compress_kv
        attention_pe_mode = config.attention_pe_mode
        pt_input_size = config.pt_input_size
        pt_num_frames = config.pt_num_frames
        intp_vfreq = config.intp_vfreq

        self.config = config

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.compress_kv = compress_kv
        self.gradient_checkpointing = False

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        assert self.extras == 1 or self.extras == 2
        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size = hidden_size

        if pt_input_size is None:
            pt_input_size = input_size
        if pt_num_frames is None:
            pt_num_frames = num_frames
        self.blocks = []
        for i in range(depth):
            if i % 2 == 0:
                m = TransformerBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode,
                    attention_pe_mode=attention_pe_mode,
                    hw=(input_size[0] // patch_size, input_size[1] // patch_size),
                    pt_hw=(pt_input_size[0] // patch_size, pt_input_size[1] // patch_size),
                    intp_vfreq=intp_vfreq, compress_kv=compress_kv
                )
            else:
                m = TransformerBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode,
                    attention_pe_mode=attention_pe_mode,
                    hw=(num_frames, 1),
                    pt_hw=(pt_num_frames, 1),
                    intp_vfreq=intp_vfreq, compress_kv=compress_kv
                )
            self.blocks.append(m)
        self.blocks = nn.ModuleList(self.blocks)

        self.final_layer = FinalLayer(hidden_size, patch_size_t, patch_size, self.out_channels)
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

        x = rearrange(x, 'b f c h w -> (b f) c h w')

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)

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


#################################################################################
#                                   Latte Configs                                  #
#################################################################################

from .configuration_latte import (
    Latte_XL_122_Config, Latte_XL_144_Config, Latte_XL_188_Config,
    Latte_L_122_Config, Latte_L_144_Config, Latte_L_188_Config,
    Latte_B_122_Config, Latte_B_144_Config, Latte_B_188_Config,
    Latte_S_122_Config, Latte_S_144_Config, Latte_S_188_Config, LatteConfiguration,
)

def Latte_XL_122(**kwargs):
    return Latte(Latte_XL_122_Config(**kwargs))

def Latte_XL_144(**kwargs):
    return Latte(Latte_XL_144_Config(**kwargs))

def Latte_XL_188(**kwargs):
    return Latte(Latte_XL_188_Config(**kwargs))

def Latte_L_122(**kwargs):
    return Latte(Latte_L_122_Config(**kwargs))

def Latte_L_144(**kwargs):
    return Latte(Latte_L_144_Config(**kwargs))

def Latte_L_188(**kwargs):
    return Latte(Latte_L_188_Config(**kwargs))

def Latte_B_122(**kwargs):
    return Latte(Latte_B_122_Config(**kwargs))

def Latte_B_144(**kwargs):
    return Latte(Latte_B_144_Config(**kwargs))

def Latte_B_188(**kwargs):
    return Latte(Latte_B_188_Config(**kwargs))

def Latte_S_122(**kwargs):
    return Latte(Latte_S_122_Config(**kwargs))

def Latte_S_144(**kwargs):
    return Latte(Latte_S_144_Config(**kwargs))

def Latte_S_188(**kwargs):
    return Latte(Latte_S_188_Config(**kwargs))


Latte_models = {
    "Latte-XL/122": Latte_XL_122, "Latte-XL/144": Latte_XL_144, "Latte-XL/188": Latte_XL_188,
    "Latte-L/122": Latte_L_122, "Latte-L/144": Latte_L_144, "Latte-L/188": Latte_L_188,
    "Latte-B/122": Latte_B_122, "Latte-B/144": Latte_B_144, "Latte-B/188": Latte_B_188,
    "Latte-S/122": Latte_S_122, "Latte-S/144": Latte_S_144, "Latte-S/188": Latte_S_188,
}


if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    network = Latte_XL_122().to(device)
    from thop import profile 
    flops, params = profile(network, inputs=(img, t))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    # out = network(img, t, y)
    # print(out.shape)
