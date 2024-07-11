import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
from typing import Optional, Dict, Any
from einops import rearrange, repeat

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None

from .modeling_latte import LatteT2V
from .modeling_latte import Transformer3DModelOutput
from .modules import PatchEmbed
from diffusers.configuration_utils import register_to_config 


def hook_forward_fn(module, input, output):
    print("It's forward: ")
    print(f"module: {module}")
    print("="*20)

def hook_backward_fn(module, grad_input, grad_output):
    print("It's backward: ")
    print(f"module: {module}")
    print(f"grad_input is None?: {grad_input is None}")
    print(grad_input)
    print(f"grad_output is None?: {grad_output is None}")
    print(grad_output)
    print("="*20)


def hacked_forward_for_vip(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        vip_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        vip_attention_mask : Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        
        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num  # 20-4=16
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

        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            # NOTE add vip attention mask
            encoder_attention_mask = torch.cat([encoder_attention_mask, vip_attention_mask], dim=1) # B N -> B N+num_vip_tokens

            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = repeat(encoder_attention_mask, 'b 1 l -> (b f) 1 l', f=frame).contiguous()
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            # NOTE add vip attention mask
            encoder_attention_mask = torch.cat([encoder_attention_mask, vip_attention_mask], dim=2) # B 1+image_num N -> B 1+image_num N+num_vip_tokens

            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(encoder_attention_mask_video, 'b 1 l -> b (1 f) l',
                                                  f=frame).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)
            encoder_attention_mask = rearrange(encoder_attention_mask, 'b n l -> (b n) l').contiguous().unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        if npu_config is not None:
            encoder_attention_mask = npu_config.get_attention_mask(encoder_attention_mask, attention_mask.shape[-2])

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hw = (height, width)
            num_patches = height * width

            hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings

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
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  
            # NOTE add vip hidden states
            encoder_hidden_states = torch.cat([encoder_hidden_states, vip_hidden_states], dim=2)  # # B 1+image_num N D -> B 1+image_num N+num_vip_tokens D

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(encoder_hidden_states_video, 'b 1 t d -> b (1 f) t d', f=frame).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
                encoder_hidden_states_spatial = rearrange(encoder_hidden_states, 'b f t d -> (b f) t d').contiguous()
            else:
                encoder_hidden_states_spatial = repeat(encoder_hidden_states, 'b 1 t d -> (b f) t d', f=frame).contiguous()
        else:
            # NOTE add vip hidden states
            encoder_hidden_states = torch.cat([encoder_hidden_states, vip_hidden_states], dim=2)  # # B 1+image_num N D -> B 1+image_num N+num_vip_tokens D
        

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
