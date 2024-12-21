import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple, List
from torch.nn import functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm, FP32LayerNorm
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.opensora_t2i.modules import CombinedTimestepTextProjEmbeddings, BasicTransformerBlock, AdaNorm
from opensora.utils.utils import to_2tuple
from opensora.models.diffusion.common import PatchEmbed2D
from opensora.models.diffusion.opensora_t2i.modules import RoPE3D, PositionGetter3D, FlashFP32LayerNorm
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info



                

class OpenSoraT2I(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: List[int] = [2, 4, 8, 4, 2], 
        sparse_n: List[int] = [1, 4, 16, 4, 1], 
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        caption_channels_2: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        pooled_projection_dim: int = 1024, 
        timestep_embed_dim: int = 512,
        norm_cls: str = 'fp32_layer_norm', 
        skip_connection: bool = False, 
        norm_skip: bool = False, 
        explicit_uniform_rope: bool = False, 
        layerwise_text_mlp: bool = False,
        time_as_x_token: bool = False,
        time_as_text_token: bool = False,
        sandwich_norm: bool = False,
        conv_out: bool = False,
        conv_ffn: bool = False,
    ):
        super().__init__()
        assert not (time_as_x_token and time_as_text_token), "Cannot have both time_as_token and time_as_text_token"
        # Set some common variables used across the board.
        self.out_channels = in_channels if out_channels is None else out_channels
        self.config.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'fp32_layer_norm':
            self.norm_cls = FP32LayerNorm

        assert len(self.config.num_layers) % 2 == 1

        self._init_patched_inputs()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _init_patched_inputs(self):

        # 0. some param
        interpolation_scale_thw = (
            self.config.interpolation_scale_t, 
            self.config.interpolation_scale_h, 
            self.config.interpolation_scale_w
            )
        
        # 1. patch embedding
        self.patch_embed = PatchEmbed2D(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )
        
        # 2. time embedding and pooled text embedding
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            timestep_embed_dim=self.config.timestep_embed_dim, 
            embedding_dim=self.config.timestep_embed_dim, 
            pooled_projection_dim=self.config.pooled_projection_dim, 
            time_as_token=self.config.time_as_x_token or self.config.time_as_text_token
        )

        # 3. anthor text embedding
        text_hidden_size = self.config.caption_channels
        self.caption_projection_2 = None
        if self.config.caption_channels_2 is not None and self.config.caption_channels_2 > 0:
            text_hidden_size = max(text_hidden_size, self.config.caption_channels_2)
            self.caption_projection_2 = PixArtAlphaTextProjection(self.config.caption_channels_2, text_hidden_size, act_fn="silu_fp32")
        self.caption_projection = PixArtAlphaTextProjection(self.config.caption_channels, text_hidden_size, act_fn="silu_fp32")

        # 4. rope
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D(
            self.config.sample_size_t, self.config.sample_size_h, self.config.sample_size_w, 
            self.config.explicit_uniform_rope
            )

        # forward transformer blocks
        self.transformer_blocks = []
        if self.config.skip_connection:
            assert len(self.config.num_layers) == 3
            assert self.config.num_layers[0] == self.config.num_layers[2]
            self.skip_norm_linear = [
                nn.Sequential(
                    self.norm_cls(
                        self.config.hidden_size*2, 
                        elementwise_affine=self.config.norm_elementwise_affine, 
                        eps=self.config.norm_eps
                        ) if self.config.norm_skip else nn.Identity(), 
                    nn.Linear(self.config.hidden_size*2, self.config.hidden_size), 
                ) for _ in range(self.config.num_layers[0])
            ]

        for idx, num_layer in enumerate(self.config.num_layers):
            stage_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        self.config.hidden_size,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        self.config.timestep_embed_dim, 
                        text_hidden_size, 
                        dropout=self.config.dropout,
                        activation_fn=self.config.activation_fn,
                        attention_bias=self.config.attention_bias,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                        interpolation_scale_thw=interpolation_scale_thw, 
                        norm_cls=self.config.norm_cls, 
                        layerwise_text_mlp=self.config.layerwise_text_mlp, 
                        time_as_x_token=self.config.time_as_x_token, 
                        time_as_text_token=self.config.time_as_text_token, 
                        sandwich_norm=self.config.sandwich_norm, 
                        conv_ffn=self.config.conv_ffn
                    )
                    for i in range(num_layer)
                ]
            )
            self.transformer_blocks.append(stage_blocks)
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        if self.config.skip_connection:
            self.skip_norm_linear = nn.ModuleList(self.skip_norm_linear)

        # norm final and unpatchfy
        self.norm_final = FP32LayerNorm(
            self.config.hidden_size, eps=self.config.norm_eps, 
            elementwise_affine=self.config.norm_elementwise_affine
            )
        self.proj_out = nn.Linear(
            self.config.hidden_size, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
        )
        self.final_conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1) if self.config.conv_out else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs, 
    ):
        
        batch_size, c, frame, h, w = hidden_states.shape
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
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame, h, w -> a video with images
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.max_pool3d(
                attention_mask, 
                kernel_size=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size), 
                stride=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size)
                )
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)') 
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)


        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if encoder_attention_mask_2 is not None and encoder_attention_mask_2.ndim == 3 and self.caption_projection_2 is not None:  
            # b, 1, l -> only images
            encoder_attention_mask_2 = (1 - encoder_attention_mask_2.to(self.dtype)) * -10000.0
            encoder_attention_mask_2 = encoder_attention_mask_2.unsqueeze(1)

            encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_attention_mask_2], dim=-1)

        # 1. Input
        frame = ((frame - 1) // self.config.patch_size_t + 1) if frame % 2 == 1 else frame // self.config.patch_size_t  # patchfy
        height, width = hidden_states.shape[-2] // self.config.patch_size, hidden_states.shape[-1] // self.config.patch_size


        hidden_states, encoder_hidden_states, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, encoder_hidden_states_2, timestep, pooled_projections
        )
        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()

        # 2. Blocks
        pos_thw = self.position_getter(
            batch_size, t=frame, h=height, w=width, 
            device=hidden_states.device, training=self.training
            )
        video_rotary_emb = self.rope(self.attention_head_dim, pos_thw, hidden_states.device)

        hidden_states, skip_connections = self._operate_on_enc(
            hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, 
            embedded_timestep, video_rotary_emb, frame, height, width
            )
        
        hidden_states = self._operate_on_mid(
            hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, 
            embedded_timestep, video_rotary_emb, frame, height, width
            )
        
        hidden_states = self._operate_on_dec(
            hidden_states, skip_connections, encoder_hidden_states, attention_mask, encoder_attention_mask, 
            embedded_timestep, video_rotary_emb, frame, height, width
            )

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states, frame, height=height, width=width,
        )  # b c t h w

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _operate_on_enc(
            self, hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, 
            embedded_timestep, video_rotary_emb, frame, height, width
        ):
        
        skip_connections = []
        for idx, stage_block in enumerate(self.transformer_blocks[:len(self.config.num_layers)//2]):
            for idx_, block in enumerate(stage_block):
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask, 
                        embedded_timestep,
                        video_rotary_emb, 
                        frame, 
                        height, 
                        width, 
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask, 
                        embedded_timestep=embedded_timestep,
                        video_rotary_emb=video_rotary_emb, 
                        frame=frame, 
                        height=height, 
                        width=width, 
                    )
                if self.config.skip_connection:
                    skip_connections.append(hidden_states)
        # import sys;sys.exit()
        return hidden_states, skip_connections

    def _operate_on_mid(
            self, hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, 
            embedded_timestep, video_rotary_emb, frame, height, width
        ):
        
        for idx_, block in enumerate(self.transformer_blocks[len(self.config.num_layers)//2]):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask, 
                    embedded_timestep,
                    video_rotary_emb, 
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask, 
                    embedded_timestep=embedded_timestep,
                    video_rotary_emb=video_rotary_emb, 
                    frame=frame, 
                    height=height, 
                    width=width, 
                )
        return hidden_states


    def _operate_on_dec(
            self, hidden_states, skip_connections, encoder_hidden_states, attention_mask, encoder_attention_mask, 
            embedded_timestep, video_rotary_emb, frame, height, width
        ):
        
        for idx, stage_block in enumerate(self.transformer_blocks[-(len(self.config.num_layers)//2):]):
            for idx_, block in enumerate(stage_block):
                if self.config.skip_connection:
                    skip_hidden_states = skip_connections.pop()
                    hidden_states = torch.cat([hidden_states, skip_hidden_states], dim=-1)
                    hidden_states = self.skip_norm_linear[idx_](hidden_states)

                if self.training and self.gradient_checkpointing:
                    
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask, 
                        embedded_timestep,
                        video_rotary_emb, 
                        frame, 
                        height, 
                        width, 
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask, 
                        embedded_timestep=embedded_timestep,
                        video_rotary_emb=video_rotary_emb, 
                        frame=frame, 
                        height=height, 
                        width=width, 
                    )
        return hidden_states


    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, encoder_hidden_states_2, timestep, pooled_projections):
        
        # print('_operate_on_patched_inputs')
        # print(f'enc hidden_states, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'enc encoder_hidden_states, ', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        hidden_states = self.patch_embed(hidden_states.to(self.dtype))
        assert pooled_projections is None or pooled_projections.shape[1] == 1
        if pooled_projections is not None:
            pooled_projections = pooled_projections.squeeze(1)  # b 1 1 d -> b 1 d
        timesteps_emb = self.time_text_embed(timestep, pooled_projections, hidden_states.dtype)  # (N, D)
            
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d
        assert encoder_hidden_states.shape[1] == 1  # b, 1, l, d
        encoder_hidden_states = encoder_hidden_states.squeeze(1)

        if encoder_hidden_states_2 is not None and self.caption_projection_2 is not None:
            encoder_hidden_states_2 = self.caption_projection_2(encoder_hidden_states_2)  # b, 1, l, d
            assert encoder_hidden_states_2 is None or encoder_hidden_states_2.shape[1] == 1  # b, 1, l, d
            encoder_hidden_states_2 = encoder_hidden_states_2.squeeze(1)

            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=1)
        # print('_operate_on_patched_inputs')
        # print(f'enc timesteps_emb, ', 
        #         f'max {timesteps_emb.max()}, min {timesteps_emb.min()}, mean {timesteps_emb.mean()}, std {timesteps_emb.std()}')
        # print(f'enc hidden_states, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'enc encoder_hidden_states, ', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        # print('-----------------------')
        return hidden_states, encoder_hidden_states, timesteps_emb
    
    def _get_output_for_patched_inputs(
        self, hidden_states, num_frames, height, width
    ):  
        hidden_states = self.norm_final(hidden_states)
        # unpatchify
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            -1, num_frames, height, width, 
            self.config.patch_size_t, self.config.patch_size, self.config.patch_size, self.out_channels
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            -1, 
            self.out_channels, 
            num_frames * self.config.patch_size_t, 
            height * self.config.patch_size, 
            width * self.config.patch_size
        )
        if self.final_conv is not None:
            output = rearrange(output, "b c t h w -> (b t) c h w", t=num_frames * self.config.patch_size_t)
            output = self.final_conv(output)
            output = rearrange(output, "(b t) c h w -> b c t h w", t=num_frames * self.config.patch_size_t)
        return output


def OpenSoraT2I_2B_111(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=1, 
        caption_channels=4096, pooled_projection_dim=0, **kwargs
    )

def OpenSoraT2I_2B_122(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, **kwargs
    )

def OpenSoraT2I_2B_122_4k3584(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        caption_channels_2=3584, **kwargs
    )

def OpenSoraT2I_2B_122_4k4k(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        caption_channels_2=4096, **kwargs
    )

def OpenSoraT2I_2B_122_SandWichNorm(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        sandwich_norm=True, **kwargs
    )

def OpenSoraT2I_2B_122_ConvFFN(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        conv_ffn=True, **kwargs
    )
def OpenSoraT2I_2B_122_RMSN(**kwargs):
    kwargs.pop('norm_cls', None)
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, norm_cls='rms_norm', **kwargs
    )

def OpenSoraT2I_2B_122_FinalConv(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        conv_out=True, **kwargs
    )
def OpenSoraT2I_2B_122_Skip(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        skip_connection=True, **kwargs
    )

def OpenSoraT2I_2B_122_Norm_Skip(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        skip_connection=True, norm_skip=True, **kwargs
    )


def OpenSoraT2I_2B_122_CLIP(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=1280, **kwargs
    )

def OpenSoraT2I_2B_122_TextMLP(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        layerwise_text_mlp=True, **kwargs
    )


def OpenSoraT2I_2B_122_TimeAsX(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        time_as_x_token=True, **kwargs
    )

def OpenSoraT2I_2B_122_TimeAsT(**kwargs):
    return OpenSoraT2I(  # 33 layers
        num_layers=[16, 1, 16], 
        attention_head_dim=64, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=4096, pooled_projection_dim=0, 
        time_as_text_token=True, **kwargs
    )

OpenSora_T2I_models = {
    "OpenSoraT2I-2B/111": OpenSoraT2I_2B_111, 
    "OpenSoraT2I-2B/122": OpenSoraT2I_2B_122, 
    "OpenSoraT2I-2B/122/4k3584": OpenSoraT2I_2B_122_4k3584, 
    "OpenSoraT2I-2B/122/4k4k": OpenSoraT2I_2B_122_4k4k, 
    "OpenSoraT2I-2B/122/ConvFFN": OpenSoraT2I_2B_122_ConvFFN, 
    "OpenSoraT2I-2B/122/FinalConv": OpenSoraT2I_2B_122_FinalConv, 
    "OpenSoraT2I-2B/122/SandWichNorm": OpenSoraT2I_2B_122_SandWichNorm, 
    "OpenSoraT2I-2B/122/RMSN": OpenSoraT2I_2B_122_RMSN, 
    "OpenSoraT2I-2B/122/Skip": OpenSoraT2I_2B_122_Skip, 
    "OpenSoraT2I-2B/122/Norm_Skip": OpenSoraT2I_2B_122_Norm_Skip, 
    "OpenSoraT2I-2B/122/CLIP": OpenSoraT2I_2B_122_CLIP, 
    "OpenSoraT2I-2B/122/TextMLP": OpenSoraT2I_2B_122_TextMLP, 
    "OpenSoraT2I-2B/122/TimeAsX": OpenSoraT2I_2B_122_TimeAsX, 
    "OpenSoraT2I-2B/122/TimeAsT": OpenSoraT2I_2B_122_TimeAsT, 
}

OpenSora_T2I_models_class = {
    "OpenSoraT2I-2B/111": OpenSoraT2I,
    "OpenSoraT2I-2B/122": OpenSoraT2I,
    "OpenSoraT2I-2B/122/4k3584": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/4k4k": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/ConvFFN": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/FinalConv": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/SandWichNorm": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/RMSN": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/Skip": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/Norm_Skip": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/CLIP": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/TextMLP": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/TimeAsX": OpenSoraT2I, 
    "OpenSoraT2I-2B/122/TimeAsT": OpenSoraT2I, 
}

if __name__ == '__main__':
    '''
    python opensora/models/diffusion/opensora_v1_5/modeling_opensora.py
    '''
    from deepspeed.runtime.utils import get_weight_norm
    from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
    from opensora.models import CausalVAEModelWrapper
    from opensora.utils.ema_utils import EMAModel
    from opensora.utils.deepspeed_utils import get_weight_norm_dict
    args = type('args', (), 
    {
        'ae': 'WFVAEModel_D32_8x8x8', 
        'model_max_length': 300, 
        'max_height': 640,
        'max_width': 640,
        'num_frames': 105,
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        "sparse1d": False, 
        "rank": 64, 
    }
    )
    b = 1
    c = 32
    cond_c = 4096
    cond_c2 = 4096
    cond_c3 = 1280
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size_h = args.max_height // ae_stride_h
    latent_size_w = args.max_width // ae_stride_w
    num_frames = (args.num_frames - 1) // ae_stride_t + 1
    

    # model = OpenSoraT2I.from_pretrained("11.10_mmdit13b_dense_rf_bs8192_lr5e-5_max1x384x384_min1x384x288_emaclip99_border109m/checkpoint-5967/model_ema")
    # for blk in model.transformer_blocks:
    #     for i in blk:
    #         weight_norm = get_weight_norm(parameters=i.parameters(), mpu=None)
    #         print(weight_norm)
    # state_dict = model.state_dict()
    
    # device = torch.device('cpu')
    device = torch.device('cuda:0')

    model = OpenSoraT2I_2B_122_ConvFFN(
        in_channels=c, 
        out_channels=c, 
        sample_size_h=latent_size_h, 
        sample_size_w=latent_size_w, 
        sample_size_t=num_frames, 
        interpolation_scale_t=args.interpolation_scale_t, 
        interpolation_scale_h=args.interpolation_scale_h, 
        interpolation_scale_w=args.interpolation_scale_w, 
        )
    print(model)
    # model_.load_state_dict(state_dict, strict=False)
    # model_.save_pretrained('12.11_14bmmdit_final384_rms2layer')
    # import sys;sys.exit()
    # import ipdb;ipdb.set_trace()
    weight_norm_dict = get_weight_norm_dict(model)
    import ipdb;ipdb.set_trace()
    # print(weight_norm)
    # import sys;sys.exit()
    total_cnt = len(list(model.named_parameters()))
    # print('total_cnt', total_cnt)
    # print(k, 'total_cnt', total_cnt)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    # import sys;sys.exit()
    # try:
        # path = "/storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/debug/checkpoint-10/pytorch_model.bin"
        # ckpt = torch.load(path, map_location="cpu")
        # msg = model.load_state_dict(ckpt, strict=True) # OpenSoraT2V_v1_5.from_pretrained('/storage/ongoing/9.29/mmdit/1.5/Open-Sora-Plan/test_ckpt')
        
        # ema_model = EMAModel.from_pretrained('./test_v1_5', OpenSoraT2V_v1_5)
        # ema_model.save_pretrained('./test_v1_5_ema')
        # with open('config.json', "w", encoding="utf-8") as writer:
        #     writer.write(model.to_json_string())
    #     print(msg)
    # except Exception as e:
    #     print(e)
    model = model.to(device)
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1, args.model_max_length, cond_c).to(device)
    cond_2 = torch.randn(b, 1, args.model_max_length, cond_c2).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1, args.model_max_length)).to(device)  # B 1 L
    cond_mask_2 = torch.randint(0, 2, (b, 1, args.model_max_length)).to(device)  # B 1 L
    timestep = torch.randint(0, 1000, (b,), device=device)
    pooled_projections = torch.randn(b, 1, cond_c3).to(device)
    model_kwargs = dict(hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask, 
                        pooled_projections=pooled_projections, 
                        encoder_attention_mask=cond_mask, timestep=timestep, 
                        encoder_hidden_states_2=cond_2, encoder_attention_mask_2=cond_mask_2)
    with torch.no_grad():
        output = model(**model_kwargs)
    print(output[0].shape)
    # model.save_pretrained('./test_v1_5_6b')

