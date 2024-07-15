import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
from typing import Optional, Dict, Any
from einops import rearrange, repeat

from diffusers.utils import is_torch_version, deprecate

try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None

from .modeling_opensora import OpenSoraT2V
from .modeling_opensora import Transformer2DModelOutput

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


def hacked_model(model):
    model._operate_on_patched_inputs = operate_on_patched_inputs.__get__(model, OpenSoraT2V)
    model.forward = hacked_forward_for_vip.__get__(model, OpenSoraT2V)


def operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, vip_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num):
    # batch_size = hidden_states.shape[0]
    hidden_states_vid, hidden_states_img = self.pos_embed(hidden_states.to(self.dtype), frame)
    timestep_vid, timestep_img = None, None
    embedded_timestep_vid, embedded_timestep_img = None, None
    encoder_hidden_states_vid, encoder_hidden_states_img = None, None

    if self.adaln_single is not None:
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d
        if hidden_states_vid is None:
            timestep_img = timestep
            embedded_timestep_img = embedded_timestep
        else:
            timestep_vid = timestep
            embedded_timestep_vid = embedded_timestep
            if hidden_states_img is not None:
                timestep_img = repeat(timestep, 'b d -> (b i) d', i=use_image_num).contiguous()
                embedded_timestep_img = repeat(embedded_timestep, 'b d -> (b i) d', i=use_image_num).contiguous()

    if self.caption_projection is not None:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d

        # NOTE add vip hidden states
        encoder_hidden_states = torch.cat([encoder_hidden_states, vip_hidden_states], dim=2)  # # B 1+image_num N D -> B 1+image_num N+num_vip_tokens D

        if hidden_states_vid is None:
            encoder_hidden_states_img = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')
        else:
            encoder_hidden_states_vid = rearrange(encoder_hidden_states[:, :1], 'b 1 l d -> (b 1) l d')
            if hidden_states_img is not None:
                encoder_hidden_states_img = rearrange(encoder_hidden_states[:, 1:], 'b i l d -> (b i) l d')


    return hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img

    

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
    use_image_num: Optional[int] = 0,
    return_dict: bool = True,
):
    
    
    batch_size, c, frame, h, w = hidden_states.shape
    frame = frame - use_image_num  # 21-4=17
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            print.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
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
    attention_mask_vid, attention_mask_img = None, None
    if attention_mask is not None and attention_mask.ndim == 4:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #   (keep = +0,     discard = -10000.0)
        # b, frame+use_image_num, h, w -> a video with images
        # b, 1, h, w -> only images
        attention_mask = attention_mask.to(self.dtype)
        if npu_config is not None and get_sequence_parallel_state():
            attention_mask_vid = attention_mask[:, :frame * hccl_info.world_size]  # b, frame, h, w
            attention_mask_img = attention_mask[:, frame * hccl_info.world_size:]  # b, use_image_num, h, w
        else:
            attention_mask_vid = attention_mask[:, :frame]  # b, frame, h, w
            attention_mask_img = attention_mask[:, frame:]  # b, use_image_num, h, w

        if attention_mask_vid.numel() > 0:
            attention_mask_vid_first_frame = attention_mask_vid[:, :1].repeat(1, self.patch_size_t-1, 1, 1)
            attention_mask_vid = torch.cat([attention_mask_vid_first_frame, attention_mask_vid], dim=1)
            attention_mask_vid = attention_mask_vid.unsqueeze(1)  # b 1 t h w
            attention_mask_vid = F.max_pool3d(attention_mask_vid, kernel_size=(self.patch_size_t, self.patch_size, self.patch_size), 
                                                stride=(self.patch_size_t, self.patch_size, self.patch_size))
            attention_mask_vid = rearrange(attention_mask_vid, 'b 1 t h w -> (b 1) 1 (t h w)') 
        if attention_mask_img.numel() > 0:
            attention_mask_img = F.max_pool2d(attention_mask_img, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
            attention_mask_img = rearrange(attention_mask_img, 'b i h w -> (b i) 1 (h w)') 

        attention_mask_vid = (1 - attention_mask_vid.bool().to(self.dtype)) * -10000.0 if attention_mask_vid.numel() > 0 else None
        attention_mask_img = (1 - attention_mask_img.bool().to(self.dtype)) * -10000.0 if attention_mask_img.numel() > 0 else None

        if frame == 1 and use_image_num == 0:
            attention_mask_img = attention_mask_vid
            attention_mask_vid = None
    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    # import ipdb;ipdb.set_trace()
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  

        # b, 1+use_image_num, l -> a video with images
        # b, 1, l -> only images
        # NOTE add vip attention mask
        encoder_attention_mask = torch.cat([encoder_attention_mask, vip_attention_mask], dim=-1) # B 1+image_num N -> B 1+image_num N+num_vip_tokens

        encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
        in_t = encoder_attention_mask.shape[1]
        encoder_attention_mask_vid = encoder_attention_mask[:, :in_t-use_image_num]  # b, 1, l
        encoder_attention_mask_vid = rearrange(encoder_attention_mask_vid, 'b 1 l -> (b 1) 1 l') if encoder_attention_mask_vid.numel() > 0 else None

        encoder_attention_mask_img = encoder_attention_mask[:, in_t-use_image_num:]  # b, use_image_num, l
        encoder_attention_mask_img = rearrange(encoder_attention_mask_img, 'b i l -> (b i) 1 l') if encoder_attention_mask_img.numel() > 0 else None

        if frame == 1 and use_image_num == 0:
            encoder_attention_mask_img = encoder_attention_mask_vid
            encoder_attention_mask_vid = None

    if npu_config is not None and attention_mask_vid is not None:
        attention_mask_vid = npu_config.get_attention_mask(attention_mask_vid, attention_mask_vid.shape[-1])
        encoder_attention_mask_vid = npu_config.get_attention_mask(encoder_attention_mask_vid,
                                                                    attention_mask_vid.shape[-2])
    if npu_config is not None and attention_mask_img is not None:
        attention_mask_img = npu_config.get_attention_mask(attention_mask_img, attention_mask_img.shape[-1])
        encoder_attention_mask_img = npu_config.get_attention_mask(encoder_attention_mask_img,
                                                                    attention_mask_img.shape[-2])


    # 1. Input
    frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
    height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, \
    timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
        hidden_states, encoder_hidden_states, vip_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
    )
    # 2. Blocks
    # import ipdb;ipdb.set_trace()
    if npu_config is not None and get_sequence_parallel_state():
        if hidden_states_vid is not None:
            hidden_states_vid = rearrange(hidden_states_vid, 'b s h -> s b h', b=batch_size).contiguous()
            encoder_hidden_states_vid = rearrange(encoder_hidden_states_vid, 'b s h -> s b h',
                                                    b=batch_size).contiguous()
            timestep_vid = timestep_vid.view(batch_size, 6, -1).transpose(0, 1).contiguous()
    for block in self.transformer_blocks:
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            # import ipdb;ipdb.set_trace()
            if hidden_states_vid is not None:
                hidden_states_vid = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states_vid,
                    attention_mask_vid,
                    encoder_hidden_states_vid,
                    encoder_attention_mask_vid,
                    timestep_vid,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
            # import ipdb;ipdb.set_trace()
            if hidden_states_img is not None:
                hidden_states_img = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states_img,
                    attention_mask_img,
                    encoder_hidden_states_img,
                    encoder_attention_mask_img,
                    timestep_img,
                    cross_attention_kwargs,
                    class_labels,
                    1, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
        else:
            if hidden_states_vid is not None:
                hidden_states_vid = block(
                    hidden_states_vid,
                    attention_mask=attention_mask_vid,
                    encoder_hidden_states=encoder_hidden_states_vid,
                    encoder_attention_mask=encoder_attention_mask_vid,
                    timestep=timestep_vid,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )
            if hidden_states_img is not None:
                hidden_states_img = block(
                    hidden_states_img,
                    attention_mask=attention_mask_img,
                    encoder_hidden_states=encoder_hidden_states_img,
                    encoder_attention_mask=encoder_attention_mask_img,
                    timestep=timestep_img,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=1, 
                    height=height, 
                    width=width, 
                )

    if npu_config is not None and get_sequence_parallel_state():
        if hidden_states_vid is not None:
            hidden_states_vid = rearrange(hidden_states_vid, 's b h -> b s h', b=batch_size).contiguous()

    # 3. Output
    output_vid, output_img = None, None 
    if hidden_states_vid is not None:
        output_vid = self._get_output_for_patched_inputs(
            hidden_states=hidden_states_vid,
            timestep=timestep_vid,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep_vid,
            num_frames=frame, 
            height=height,
            width=width,
        )  # b c t h w
    if hidden_states_img is not None:
        output_img = self._get_output_for_patched_inputs(
            hidden_states=hidden_states_img,
            timestep=timestep_img,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep_img,
            num_frames=1, 
            height=height,
            width=width,
        )  # b c 1 h w
        if use_image_num != 0:
            output_img = rearrange(output_img, '(b i) c 1 h w -> b c i h w', i=use_image_num)

    if output_vid is not None and output_img is not None:
        output = torch.cat([output_vid, output_img], dim=2)
    elif output_vid is not None:
        output = output_vid
    elif output_img is not None:
        output = output_img

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

