
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
from typing import Optional, Dict, Any
from einops import rearrange, repeat

from .modeling_latte import LatteT2V
from .modeling_latte import Transformer3DModelOutput
from .modules import PatchEmbed

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class OpenSoraInpaint(LatteT2V):
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
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            attention_mode: str = 'flash', 
            use_rope: bool = False, 
            model_max_length: int = 300, 
            rope_scaling_type: str = 'linear', 
            compress_kv_factor: int = 1, 
            interpolation_scale_h: float = None,
            interpolation_scale_w: float = None,
            interpolation_scale_t: float = None,
            downsampler: str = None,
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
            attention_mode=attention_mode,
            use_rope=use_rope,
            model_max_length=model_max_length,
            rope_scaling_type=rope_scaling_type,
            compress_kv_factor=compress_kv_factor,
            interpolation_scale_h=interpolation_scale_h,
            interpolation_scale_w=interpolation_scale_w,
            interpolation_scale_t=interpolation_scale_t,
            downsampler=downsampler,
        )

        inner_dim = num_attention_heads * attention_head_dim
        interpolation_scale = self.interpolation_scale

        self.pos_embed_masked_video = nn.ModuleList(
            [
                PatchEmbed(
                    height=sample_size[0],
                    width=sample_size[1],
                    patch_size=patch_size,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    interpolation_scale=interpolation_scale,
                ),
                zero_module(nn.Conv1d(inner_dim, inner_dim, kernel_size=1, stride=1, padding=0))
            ]
        )

        self.pos_embed_mask = nn.ModuleList(
            [
                PatchEmbed(
                    height=sample_size[0],
                    width=sample_size[1],
                    patch_size=patch_size,
                    in_channels=in_channels,
                    embed_dim=inner_dim,
                    interpolation_scale=interpolation_scale,
                ),
                zero_module(nn.Conv1d(inner_dim, inner_dim, kernel_size=1, stride=1, padding=0))
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
       
        input_batch_size, c, frame, h, w = hidden_states.shape
        assert c == self.config.in_channels * 3, f"Input hidden_states should have channel {self.config.in_channels * 3}, but got {c}."
        assert use_image_num == 0, f"Image joint training is not supported in inpainting mode."
        hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w').contiguous()
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is None:
            attention_mask = torch.ones((input_batch_size, frame+use_image_num, h, w), device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            attention_mask = attention_mask.to(hidden_states.dtype)
        attention_mask = self.vae_to_diff_mask(attention_mask, use_image_num)
        dtype = attention_mask.dtype
        attention_mask_compress = F.max_pool2d(attention_mask.float(), kernel_size=self.compress_kv_factor, stride=self.compress_kv_factor)
        attention_mask_compress = attention_mask_compress.to(dtype)

        attention_mask = self.make_attn_mask(attention_mask, frame, hidden_states.dtype)
        attention_mask_compress = self.make_attn_mask(attention_mask_compress, frame, hidden_states.dtype)

         # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = repeat(encoder_attention_mask, 'b 1 l -> (b f) 1 l', f=frame).contiguous()
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hw = (height, width)
            num_patches = height * width

            # inpaint
            in_channels = self.config.in_channels
            hidden_states, hidden_states_masked_video, hidden_states_mask = hidden_states[:, :in_channels], hidden_states[:, in_channels: 2 * in_channels], hidden_states[:, 2 * in_channels: 3 * in_channels]
            
            hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings

            hidden_states_masked_video = self.pos_embed_masked_video[0](hidden_states_masked_video.to(self.dtype)) # alrady add positional embeddings
            hidden_states_masked_video = rearrange(hidden_states_masked_video, 'b n c -> b c n').contiguous()
            hidden_states_masked_video = self.pos_embed_masked_video[1](hidden_states_masked_video) 
            hidden_states_masked_video = rearrange(hidden_states_masked_video, 'b c n -> b n c').contiguous()

            hidden_states_mask = self.pos_embed_mask[0](hidden_states_mask.to(self.dtype)) # alrady add positional embeddings
            hidden_states_mask = rearrange(hidden_states_mask, 'b n c -> b c n').contiguous()
            hidden_states_mask = self.pos_embed_mask[1](hidden_states_mask)
            hidden_states_mask = rearrange(hidden_states_mask, 'b c n -> b n c').contiguous()

            hidden_states = hidden_states + hidden_states_masked_video + hidden_states_mask
            
            
            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # batch_size = hidden_states.shape[0]
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  # 3 120 1152
            encoder_hidden_states_spatial = repeat(encoder_hidden_states, 'b 1 t d -> (b f) t d', f=frame).contiguous()

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(timestep, 'b d -> (b f) d', f=frame + use_image_num).contiguous()
        timestep_temp = repeat(timestep, 'b d -> (b p) d', p=num_patches).contiguous()
        
        pos_hw, pos_t = None, None
        if self.use_rope:
            pos_hw, pos_t = self.make_position(input_batch_size, frame, use_image_num, height, width, hidden_states.device)

        for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    attention_mask_compress if i >= self.num_layers // 2 else attention_mask, 
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    pos_hw, 
                    pos_hw, 
                    hw, 
                    use_reentrant=False,
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, '(b f) t d -> (b t) f d', b=input_batch_size).contiguous()

                    if use_image_num != 0:  # image-video joitn training
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states_video = hidden_states_video + self.temp_pos_embed

                        hidden_states_video = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t, 
                            pos_t, 
                            (frame, ), 
                            use_reentrant=False,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=input_batch_size).contiguous()

                    else:
                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states = hidden_states + self.temp_pos_embed

                        hidden_states = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t, 
                            pos_t, 
                            (frame, ), 
                            use_reentrant=False,
                        )

                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=input_batch_size).contiguous()
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    attention_mask_compress if i >= self.num_layers // 2 else attention_mask, 
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    pos_hw, 
                    pos_hw, 
                    hw, 
                )

                if enable_temporal_attentions:
                    # b c f h w, f = 16 + 4
                    hidden_states = rearrange(hidden_states, '(b f) t d -> (b t) f d', b=input_batch_size).contiguous()

                    if use_image_num != 0 and self.training:
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        # if i == 0 and not self.use_rope:
                        #     hidden_states_video = hidden_states_video + self.temp_pos_embed

                        hidden_states_video = temp_block(
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t, 
                            pos_t, 
                            (frame, ), 
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=input_batch_size).contiguous()

                    else:
                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states = hidden_states + self.temp_pos_embed

                        hidden_states = temp_block(
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            pos_t, 
                            pos_t, 
                            (frame, ), 
                        )

                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=input_batch_size).contiguous()

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame + use_image_num).contiguous()
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            output = rearrange(output, '(b f) c h w -> b c f h w', b=input_batch_size).contiguous()

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

    def custom_load_state_dict(self, pretrained_model_path):
        print(f'Loading pretrained model from {pretrained_model_path}...')
        if 'safetensors' in pretrained_model_path:
            from safetensors.torch import load_file as safe_load
            checkpoint = safe_load(pretrained_model_path, device="cpu")
        else:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')['model']
        model_state_dict = self.state_dict()
        if not 'pos_embed_masked_video.0.weight' in checkpoint:
            checkpoint['pos_embed_masked_video.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
            checkpoint['pos_embed_masked_video.0.proj.bias'] = checkpoint['pos_embed.proj.bias']
        if not 'pos_embed_mask.0.proj.weight' in checkpoint:
            checkpoint['pos_embed_mask.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
            checkpoint['pos_embed_mask.0.proj.bias'] = checkpoint['pos_embed.proj.bias']
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)

        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {len(self.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {pretrained_model_path}!')

# depth = num_layers * 2
def OpenSoraInpaint_XL_122(**kwargs):
    return OpenSoraInpaint(num_layers=28, attention_head_dim=72, num_attention_heads=16, patch_size_t=1, patch_size=2,
                    norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)

inpaint_models = {
    "OpenSoraInpaint_XL_122": OpenSoraInpaint_XL_122,
}
    

inpaint_models_class = {
    "OpenSoraInpaint_XL_122": OpenSoraInpaint,
}