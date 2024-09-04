import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from diffusers.configuration_utils import register_to_config
from opensora.models.diffusion.opensora_v1_2.modules import PatchEmbed2D
from opensora.utils.utils import to_2tuple


from opensora.models.diffusion.opensora_v1_2.modeling_opensora import OpenSoraT2V_v1_2

import glob

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def reconstitute_checkpoint(pretrained_checkpoint, model_state_dict):
    pretrained_keys = set(list(pretrained_checkpoint.keys()))
    model_keys = set(list(model_state_dict.keys()))
    common_keys = list(pretrained_keys & model_keys)
    checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
    return checkpoint


class OpenSoraInpaint(OpenSoraT2V_v1_2):
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
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale_h: float = None,
        interpolation_scale_w: float = None,
        interpolation_scale_t: float = None,
        use_additional_conditions: Optional[bool] = None,
        attention_mode: str = 'xformers', 
        downsampler: str = None, 
        use_rope: bool = False,
        use_stable_fp32: bool = False,
        sparse1d: bool = False,
        sparse2d: bool = False,
        sparse_n: int = 2,
        use_motion: bool = False,
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
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            sample_size_t=sample_size_t,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
            caption_channels=caption_channels,
            interpolation_scale_h=interpolation_scale_h,
            interpolation_scale_w=interpolation_scale_w,
            interpolation_scale_t=interpolation_scale_t,
            use_additional_conditions=use_additional_conditions,
            attention_mode=attention_mode,
            downsampler=downsampler,
            use_rope=use_rope,
            use_stable_fp32=use_stable_fp32,
            sparse1d=sparse1d,
            sparse2d=sparse2d,
            sparse_n=sparse_n,
            use_motion=use_motion,
        )

        self.vae_scale_factor_t = vae_scale_factor_t
        # init masked_pixel_values and mask conv_in
        self._init_patched_inputs_for_inpainting()

    def _init_patched_inputs_for_inpainting(self):

        assert self.config.sample_size_t is not None, "OpenSoraInpaint over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraInpaint over patched input must provide sample_size"
        #assert not (self.config.sample_size_t == 1 and self.config.patch_size_t == 2), "Image do not need patchfy in t-dim"

        self.num_frames = self.config.sample_size_t
        self.config.sample_size = to_2tuple(self.config.sample_size)
        self.height = self.config.sample_size[0]
        self.width = self.config.sample_size[1]
        self.patch_size_t = self.config.patch_size_t
        self.patch_size = self.config.patch_size
        interpolation_scale_t = ((self.config.sample_size_t - 1) // 16 + 1) if self.config.sample_size_t % 2 == 1 else self.config.sample_size_t / 16
        interpolation_scale_t = (
            self.config.interpolation_scale_t if self.config.interpolation_scale_t is not None else interpolation_scale_t
        )
        interpolation_scale = (
            self.config.interpolation_scale_h if self.config.interpolation_scale_h is not None else self.config.sample_size[0] / 30, 
            self.config.interpolation_scale_w if self.config.interpolation_scale_w is not None else self.config.sample_size[1] / 40, 
        )
        
        self.pos_embed_mask = nn.ModuleList(
            [
                PatchEmbed2D(
                    num_frames=self.config.sample_size_t,
                    height=self.config.sample_size[0],
                    width=self.config.sample_size[1],
                    patch_size_t=self.config.patch_size_t,
                    patch_size=self.config.patch_size,
                    in_channels=self.vae_scale_factor_t, # adapt for mask
                    embed_dim=self.inner_dim,
                    interpolation_scale=interpolation_scale, 
                    interpolation_scale_t=interpolation_scale_t,
                    use_abs_pos=not self.config.use_rope, 
                ),
                zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
            ]
        )
        self.pos_embed_masked_hidden_states = nn.ModuleList(
            [
                PatchEmbed2D(
                    num_frames=self.config.sample_size_t,
                    height=self.config.sample_size[0],
                    width=self.config.sample_size[1],
                    patch_size_t=self.config.patch_size_t,
                    patch_size=self.config.patch_size,
                    in_channels=self.in_channels,
                    embed_dim=self.inner_dim,
                    interpolation_scale=interpolation_scale, 
                    interpolation_scale_t=interpolation_scale_t,
                    use_abs_pos=not self.config.use_rope, 
                ),
                zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
            ]
        )

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, motion_score, batch_size, frame, use_image_num):
        # inpaint
        assert hidden_states.shape[1] == 2 * self.config.in_channels + self.vae_scale_factor_t
        in_channels = self.config.in_channels

        input_hidden_states, input_masked_hidden_states, input_mask = hidden_states[:, :in_channels], hidden_states[:, in_channels: 2 * in_channels], hidden_states[:, 2 * in_channels:]

        input_hidden_states = self.pos_embed(input_hidden_states.to(self.dtype), frame)

        input_masked_hidden_states = self.pos_embed_masked_hidden_states[0](input_masked_hidden_states.to(self.dtype), frame)
        input_masked_hidden_states = self.pos_embed_masked_hidden_states[1](input_masked_hidden_states)

        input_mask = self.pos_embed_mask[0](input_mask.to(self.dtype), frame)
        input_mask = self.pos_embed_mask[1](input_mask)

        hidden_states = input_hidden_states + input_masked_hidden_states + input_mask

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
            )  # b 6d, b d
        if self.motion_projection is not None:
            assert motion_score is not None
            motion_embed = self.motion_projection(motion_score, batch_size=batch_size, hidden_dtype=self.dtype)  # b 6d
            # print('use self.motion_projection, motion_embed:', torch.sum(motion_embed))
            timestep = timestep + motion_embed
            
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d
            assert encoder_hidden_states.shape[1] == 1
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep
    
    def transformer_model_custom_load_state_dict(self, pretrained_model_path):
        pretrained_model_path = os.path.join(pretrained_model_path, 'diffusion_pytorch_model.*')
        pretrained_model_path = glob.glob(pretrained_model_path)
        assert len(pretrained_model_path) > 0, f"Cannot find pretrained model in {pretrained_model_path}"
        pretrained_model_path = pretrained_model_path[0]

        print(f'Loading {self.__class__.__name__} pretrained weights...')
        print(f'Loading pretrained model from {pretrained_model_path}...')
        model_state_dict = self.state_dict()
        if 'safetensors' in pretrained_model_path:  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(pretrained_model_path, device="cpu")
        else:  # latest stage training weight
            pretrained_checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model' in pretrained_checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']
        checkpoint = reconstitute_checkpoint(pretrained_checkpoint, model_state_dict)

        if not 'pos_embed_masked_hidden_states.0.weight' in checkpoint:
            checkpoint['pos_embed_masked_hidden_states.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
            checkpoint['pos_embed_masked_hidden_states.0.proj.bias'] = checkpoint['pos_embed.proj.bias']

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {len(self.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {pretrained_model_path}!')

    def custom_load_state_dict(self, pretrained_model_path):
        assert isinstance(pretrained_model_path, dict), "pretrained_model_path must be a dict"

        pretrained_transformer_model_path = pretrained_model_path.get('transformer_model', None)

        self.transformer_model_custom_load_state_dict(pretrained_transformer_model_path)

def OpenSoraInpaint_S_122(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=8, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=768, **kwargs)

def OpenSoraInpaint_B_122(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraInpaint_L_122(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)

OpenSoraInpaint_models = {
    "OpenSoraInpaint-S/122": OpenSoraInpaint_S_122,  # 0.3B
    "OpenSoraInpaint-B/122": OpenSoraInpaint_B_122,  # 1.2B
    "OpenSoraInpaint-L/122": OpenSoraInpaint_L_122,  # 2.7B
}

OpenSoraInpaint_models_class = {
    "OpenSoraInpaint-S/122": OpenSoraInpaint,
    "OpenSoraInpaint-B/122": OpenSoraInpaint,
    "OpenSoraInpaint-L/122": OpenSoraInpaint,
}
