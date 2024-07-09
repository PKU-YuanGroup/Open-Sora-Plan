import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.opensora.modules import OverlapPatchEmbed3D, OverlapPatchEmbed2D, PatchEmbed2D, BasicTransformerBlock
from opensora.utils.utils import to_2tuple
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None

from .modeling_opensora import OpenSoraT2V

class OpenSoraInpaint(OpenSoraT2V):
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
        use_image_num: Optional[int] = 0,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
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
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
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


    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num):
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
            if hidden_states_vid is None:
                encoder_hidden_states_img = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')
            else:
                encoder_hidden_states_vid = rearrange(encoder_hidden_states[:, :1], 'b 1 l d -> (b 1) l d')
                if hidden_states_img is not None:
                    encoder_hidden_states_img = rearrange(encoder_hidden_states[:, 1:], 'b i l d -> (b i) l d')


        return hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img

    
    
    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, class_labels, embedded_timestep, num_frames, height=None, width=None
    ):  
        # import ipdb;ipdb.set_trace()
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=self.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, num_frames, height, width, self.patch_size_t, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, num_frames * self.patch_size_t, height * self.patch_size, width * self.patch_size)
        )
        # import ipdb;ipdb.set_trace()
        # if output.shape[2] % 2 == 0:
        #     output = output[:, :, 1:]
        return output
     
def OpenSoraT2V_S_111(**kwargs):
    return OpenSoraT2V(num_layers=28, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=1,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraT2V_B_111(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=1,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

def OpenSoraT2V_L_111(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=1,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)

def OpenSoraT2V_S_122(**kwargs):
    return OpenSoraT2V(num_layers=28, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraT2V_B_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

def OpenSoraT2V_L_122(**kwargs):
    return OpenSoraT2V(num_layers=40, attention_head_dim=128, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2048, **kwargs)

def OpenSoraT2V_ROPE_L_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)
# def OpenSoraT2V_S_222(**kwargs):
#     return OpenSoraT2V(num_layers=28, attention_head_dim=72, num_attention_heads=16, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)


def OpenSoraT2V_B_222(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=2, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

# def OpenSoraT2V_L_222(**kwargs):
#     return OpenSoraT2V(num_layers=40, attention_head_dim=128, num_attention_heads=20, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2560, **kwargs)

# def OpenSoraT2V_XL_222(**kwargs):
#     return OpenSoraT2V(num_layers=32, attention_head_dim=128, num_attention_heads=32, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=4096, **kwargs)

# def OpenSoraT2V_XXL_222(**kwargs):
#     return OpenSoraT2V(num_layers=40, attention_head_dim=128, num_attention_heads=40, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=5120, **kwargs)

OpenSora_models = {
    "OpenSoraT2V-S/122": OpenSoraT2V_S_122,  #       1.1B
    "OpenSoraT2V-B/122": OpenSoraT2V_B_122,
    "OpenSoraT2V-L/122": OpenSoraT2V_L_122,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V_ROPE_L_122,
    "OpenSoraT2V-S/111": OpenSoraT2V_S_111,
    "OpenSoraT2V-B/111": OpenSoraT2V_B_111,
    "OpenSoraT2V-L/111": OpenSoraT2V_L_111,
    # "OpenSoraT2V-XL/122": OpenSoraT2V_XL_122,
    # "OpenSoraT2V-XXL/122": OpenSoraT2V_XXL_122,
    # "OpenSoraT2V-S/222": OpenSoraT2V_S_222,
    "OpenSoraT2V-B/222": OpenSoraT2V_B_222,
    # "OpenSoraT2V-L/222": OpenSoraT2V_L_222,
    # "OpenSoraT2V-XL/222": OpenSoraT2V_XL_222,
    # "OpenSoraT2V-XXL/222": OpenSoraT2V_XXL_222,
}

OpenSora_models_class = {
    "OpenSoraT2V-S/122": OpenSoraT2V,
    "OpenSoraT2V-B/122": OpenSoraT2V,
    "OpenSoraT2V-L/122": OpenSoraT2V,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V,
    "OpenSoraT2V-S/111": OpenSoraT2V,
    "OpenSoraT2V-B/111": OpenSoraT2V,
    "OpenSoraT2V-L/111": OpenSoraT2V,

    "OpenSoraT2V-B/222": OpenSoraT2V,
}

if __name__ == '__main__':
    from opensora.models.ae import ae_channel_config, ae_stride_config
    from opensora.models.ae import getae, getae_wrapper
    from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper

    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_4x8x8', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 480,
        'max_width': 640,
        'num_frames': 61,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
    }
    )
    b = 2
    c = 8
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    if getae_wrapper(args.ae) == CausalVQVAEModelWrapper or getae_wrapper(args.ae) == CausalVAEModelWrapper:
        num_frames = (args.num_frames - 1) // ae_stride_t + 1
    else:
        num_frames = args.num_frames // ae_stride_t

    device = torch.device('cuda:0')
    model = OpenSoraT2V_L_122(in_channels=c, 
                              out_channels=c, 
                              sample_size=latent_size, 
                              sample_size_t=num_frames, 
                              activation_fn="gelu-approximate",
                            attention_bias=True,
                            attention_type="default",
                            double_self_attention=False,
                            norm_elementwise_affine=False,
                            norm_eps=1e-06,
                            norm_num_groups=32,
                            num_vector_embeds=None,
                            only_cross_attention=False,
                            upcast_attention=False,
                            use_linear_projection=False,
                            use_additional_conditions=False, 
                            downsampler=None,
                            interpolation_scale_t=args.interpolation_scale_t, 
                            interpolation_scale_h=args.interpolation_scale_h, 
                            interpolation_scale_w=args.interpolation_scale_w, 
                            use_rope=args.use_rope, 
                            ).to(device)
    
    try:
        path = "PixArt-Alpha-XL-2-512.safetensors"
        from safetensors.torch import load_file as safe_load
        ckpt = safe_load(path, device="cpu")
        # import ipdb;ipdb.set_trace()
        if ckpt['pos_embed.proj.weight'].shape != model.pos_embed.proj.weight.shape and ckpt['pos_embed.proj.weight'].ndim == 4:
            repeat = model.pos_embed.proj.weight.shape[2]
            ckpt['pos_embed.proj.weight'] = ckpt['pos_embed.proj.weight'].unsqueeze(2).repeat(1, 1, repeat, 1, 1) / float(repeat)
            del ckpt['proj_out.weight'], ckpt['proj_out.bias']
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)
    except Exception as e:
        print(e)
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    import sys;sys.exit()
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1+args.use_image_num, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1+args.use_image_num, args.model_max_length)).to(device)  # B L or B 1+num_images L
    timestep = torch.randint(0, 1000, (b,), device=device)
    model_kwargs = dict(hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask, 
                        encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, timestep=timestep)
    with torch.no_grad():
        output = model(**model_kwargs)
    print(output[0].shape)




