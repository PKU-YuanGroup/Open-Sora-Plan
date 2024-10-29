import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from diffusers.configuration_utils import register_to_config
from opensora.models.diffusion.common import PatchEmbed2D
from opensora.utils.utils import to_2tuple


from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3 as OpenSoraT2V

import glob

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class OpenSoraInpaint_v1_3(OpenSoraT2V):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        sparse_n: int = 2,
        # inpaint
        vae_scale_factor_t: int = 4,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size_h=sample_size_h,
            sample_size_w=sample_size_w,
            sample_size_t=sample_size_t,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            activation_fn=activation_fn,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            interpolation_scale_h=interpolation_scale_h,
            interpolation_scale_w=interpolation_scale_w,
            interpolation_scale_t=interpolation_scale_t,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
        )

        self.vae_scale_factor_t = vae_scale_factor_t
        # init masked_pixel_values and mask conv_in
        self._init_patched_inputs_for_inpainting()

    def _init_patched_inputs_for_inpainting(self):

        self.config.sample_size = to_2tuple(self.config.sample_size)

        self.pos_embed_masked_hidden_states = nn.ModuleList(
            [
                PatchEmbed2D(
                    patch_size=self.config.patch_size,
                    in_channels=self.config.in_channels,
                    embed_dim=self.config.hidden_size,
                ),
                zero_module(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)),
            ]
        )

        self.pos_embed_mask = nn.ModuleList(
            [
                PatchEmbed2D(
                    patch_size=self.config.patch_size,
                    in_channels=self.vae_scale_factor_t,
                    embed_dim=self.config.hidden_size,
                ),
                zero_module(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)),
            ]
        )

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, batch_size, frame):
        # inpaint
        assert hidden_states.shape[1] == 2 * self.config.in_channels + self.vae_scale_factor_t
        in_channels = self.config.in_channels

        input_hidden_states, input_masked_hidden_states, input_mask = hidden_states[:, :in_channels], hidden_states[:, in_channels: 2 * in_channels], hidden_states[:, 2 * in_channels:]

        input_hidden_states = self.pos_embed(input_hidden_states.to(self.dtype))

        input_masked_hidden_states = self.pos_embed_masked_hidden_states[0](input_masked_hidden_states.to(self.dtype))
        input_masked_hidden_states = self.pos_embed_masked_hidden_states[1](input_masked_hidden_states)

        input_mask = self.pos_embed_mask[0](input_mask.to(self.dtype))
        input_mask = self.pos_embed_mask[1](input_mask)

        hidden_states = input_hidden_states + input_masked_hidden_states + input_mask

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d or b, 1, l, d
        assert encoder_hidden_states.shape[1] == 1
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep

def OpenSoraInpaint_v1_3_2B_122(**kwargs):
   return OpenSoraInpaint_v1_3(
        num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
        caption_channels=4096, cross_attention_dim=2304, activation_fn="gelu-approximate", **kwargs
    )

OpenSoraInpaint_v1_3_models = {
    "OpenSoraInpaint_v1_3-2B/122": OpenSoraInpaint_v1_3_2B_122,  # 2.7B
}

OpenSoraInpaint_v1_3_models_class = {
    "OpenSoraInpaint_v1_3-2B/122": OpenSoraInpaint_v1_3,
    "OpenSoraInpaint_v1_3": OpenSoraInpaint_v1_3,
}
