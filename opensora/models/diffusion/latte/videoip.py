
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from opensora.models.ae.videobase.modules import attention
from opensora.models.diffusion.utils.pos_embed import RoPE1D, RoPE2D, LinearScalingRoPE2D, LinearScalingRoPE1D

from opensora.models.diffusion.latte.modules import Attention

from typing import Any, Dict, Optional, Tuple, Callable
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRACompatibleLinear



from .modules import BasicTransformerBlock
from einops import rearrange

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

@dataclass
class VideoIPOutput(BaseOutput):
    hidden_states: torch.FloatTensor
    vip_cond_mask: torch.FloatTensor


class VideoIPVideoEncoder(nn.Module):

    def __init__(
        self,
        image_encoder_type="clip",
        inner_dim=1024,
        cross_attention_dim=1152,
        num_attention_layers=2,
        use_rope=False,
        rope_scaling=None,
        attention_mode='xformers',
        vae_scale_factor_t=4,
        video_length=17,
        max_num_tokens=272,
    ):
        super().__init__()

        self.image_encoder_type = image_encoder_type
    
        self.vae_scale_factor_t = vae_scale_factor_t
        self.video_length = video_length

        self.max_num_tokens = max_num_tokens

        if USE_PEFT_BACKEND:
            linear_cls = nn.Linear
        else:
            linear_cls = LoRACompatibleLinear

        if image_encoder_type == "clip": # F * 16 * 16
            # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir="/storage/cache_dir")

            self.conv_in = nn.ModuleList(
                [
                    nn.Conv3d(in_channels=1280, out_channels=inner_dim, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),# F * 16 * 16 -> F // 2 * 16 * 16
                    nn.SiLU(),
                    nn.Conv3d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, stride=2, padding=1), # F // 2 * 16 * 16 -> F // 4 * 8 * 8
                ]
            )

            self.conv_in_downsample_factor = (4, 2, 2)
            
        elif image_encoder_type == "dino": # F * 16 * 16
            # self.image_encoder = AutoModel.from_pretrained("facebook/dinov2-giant", cache_dir="/storage/cache_dir")

            self.conv_in = nn.ModuleList(
                [
                    nn.Conv3d(in_channels=1536, out_channels=inner_dim, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),# F * 16 * 16 -> F // 2 * 16 * 16
                    nn.SiLU(),
                    nn.Conv3d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, stride=2, padding=1), # F // 2 * 16 * 16 -> F // 4 * 8 * 8
                ]
            )

            self.conv_in_downsample_factor = (4, 2, 2)

        # elif image_encoder_type == "vae": # F // 4 * 64 * 64
            # assert in_channels is not None, "Please specify latent dim for VAE"

            # self.conv_in = nn.ModuleList(
            #     [
            #         nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=2, padding=1),
            #         nn.SiLU(),
            #         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            #         nn.SiLU(),
            #         nn.Conv2d(in_channels=512, out_channels=inner_dim, kernel_size=3, stride=2, padding=1),
            #     ]
            # ) # F // 4 * 64 * 64 -> F // 4 * 8 * 8

            # self.conv_in_downsample_factor = (1, 8, 8)

        else:
            raise NotImplementedError

        self.trans_block1 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=16,
                    attention_head_dim=64,
                    dropout=0.0,
                    cross_attention_dim=None,
                    double_self_attention=False,
                    activation_fn="geglu",
                    norm_type="layer_norm",
                    use_rope=use_rope,
                    rope_scaling=rope_scaling,
                    attention_mode=attention_mode,
                )
                for _ in range(num_attention_layers)
            ]
        ) # only self-attention

        self.conv_mid = nn.Conv3d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1) # F // 4 * 8 * 8 -> F // 4 * 4 * 4
        self.conv_mid_downsample_factor = (1, 2, 2)

        self.trans_block2 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=16,
                    attention_head_dim=64,
                    dropout=0.0,
                    cross_attention_dim=None,
                    double_self_attention=False,
                    activation_fn="geglu",
                    norm_type="layer_norm",
                    use_rope=use_rope,
                    rope_scaling=rope_scaling,
                    attention_mode=attention_mode,
                )
                for _ in range(num_attention_layers)
            ] 
        ) # only self-attention

        self.proj_out = linear_cls(inner_dim, cross_attention_dim)

        self.norm_out = nn.LayerNorm(cross_attention_dim)

    def forward(
        self, 
        hidden_states,
        image_mode=False,
    ):
        # B C F H W
        input_batch_size, input_frame = hidden_states.shape[0], hidden_states.shape[2]

        if image_mode: 
            hidden_states = hidden_states.repeat_interleave(self.vae_scale_factor_t, dim=2)
            hidden_states = rearrange(hidden_states, 'b c (f i) h w -> (b f) c i h w', f=input_frame, i=self.vae_scale_factor_t)
        else:
            image_hidden_states = hidden_states[:, :, 0:1].repeat(1, 1, self.vae_scale_factor_t, 1, 1)
            hidden_states = torch.cat([image_hidden_states, hidden_states[:, :, 1:]], dim=2)
   
        for layer in self.conv_in:
            hidden_states = layer(hidden_states)

        # after conv_in
        frame, height, width = hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]
        # if training image, now batch = input_batch_size * frame; if not, batch remains the same
        hidden_states = rearrange(hidden_states, 'b c f h w -> b (f h w) c')


        for layer in self.trans_block1:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
            )
        
        # when using image mode, f=1; when using video mode, f=video_length 
        hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w ", f=frame, h=height, w=width)
        
        hidden_states = self.conv_mid(hidden_states)

        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")


        for layer in self.trans_block2:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
            )

        # when using image mode, n=1*h*w; when using video mode, n=video_length*h*w
        if image_mode:
            hidden_states = rearrange(hidden_states, '(b f) n c -> b f n c', f=input_frame)
        else:
            hidden_states = hidden_states.unsqueeze(1) # B N C -> B 1 N C


        hidden_states = self.proj_out(hidden_states)
        
        hidden_states = self.norm_out(hidden_states)

        batch, num_seq, num_tokens, _ = hidden_states.shape
        hidden_states = F.pad(hidden_states, (0, 0, 0, self.max_num_tokens - num_tokens), value=0.0)
        vip_cond_mask = torch.ones([batch, num_seq, num_tokens], device=hidden_states.device, dtype=hidden_states.dtype)
        vip_cond_mask = F.pad(vip_cond_mask, (0, self.max_num_tokens - num_tokens), value=0.0)

        # hidden_states: B 1 N D(video) B image_num N D(image), vip_cond_mask: B 1 N(video) B image_num N(image), N = max_num_tokens
        return hidden_states, vip_cond_mask

class VideoIPAdapter(nn.Module):
    def __init__(
        self,
        image_encoder_type="clip",
        inner_dim=1024,
        cross_attention_dim=1152,
        num_attention_layers=2,
        use_rope=False,
        rope_scaling=None,
        attention_mode='xformers',
        gradient_checkpointing=False,
        vae_scale_factor_t=4,
        video_length=17,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = VideoIPVideoEncoder(
            image_encoder_type=image_encoder_type,
            inner_dim=inner_dim,
            cross_attention_dim=cross_attention_dim,
            num_attention_layers=num_attention_layers,
            use_rope=use_rope,
            rope_scaling=rope_scaling,
            attention_mode=attention_mode,
            vae_scale_factor_t=vae_scale_factor_t,
            video_length=video_length,
        )

    def forward(
        self, 
        hidden_states,
        use_image_num=0,
        return_dict=True,
    ):

        assert hidden_states.ndim == 5, "Input tensor must be 5D"

        batch, channels, frame, height, width = hidden_states.shape
        frame = frame - use_image_num

        video_hidden_states = hidden_states[:, :, 0:frame]

        one_image_video = True if frame == 1 else False
        if self.training and self.gradient_checkpointing:
            video_hidden_states, video_cond_mask = torch.utils.checkpoint.checkpoint(self.encoder, video_hidden_states, one_image_video, use_reentrant=False,)
        else:
            video_hidden_states, video_cond_mask = self.encoder(video_hidden_states, image_mode=one_image_video)

        if use_image_num:
            image_hidden_states = hidden_states[:, :, frame:]
            if self.training and self.gradient_checkpointing:
                image_hidden_states, image_cond_mask = torch.utils.checkpoint.checkpoint(self.encoder, image_hidden_states, True, use_reentrant=False,)
            else:
                image_hidden_states, image_cond_mask = self.encoder(image_hidden_states, image_mode=True)
            hidden_states = torch.cat([video_hidden_states, image_hidden_states], dim=1) # B 1+image_num N D
            vip_cond_mask = torch.cat([video_cond_mask, image_cond_mask], dim=1) # B 1+image_num D
        else:
            hidden_states = video_hidden_states # B 1 N D
            vip_cond_mask = video_cond_mask # B 1 N

        
        if not return_dict:
            return (hidden_states, vip_cond_mask)
        
        return VideoIPOutput(hidden_states=hidden_states, vip_cond_mask=vip_cond_mask)


# IP Adapter
# we need use accelerate, so this processor should extend from nn.Module
class VideoIPAttnProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        dim=1152, 
        attention_mode='xformers', 
        use_rope=False, 
        rope_scaling=None, 
        compress_kv_factor=None,
        
        num_vip_tokens=272,
        vip_scale=1.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.compress_kv_factor = compress_kv_factor

        if USE_PEFT_BACKEND:
            linear_cls = nn.Linear
        else:
            linear_cls = LoRACompatibleLinear

        if self.use_rope:
            self._init_rope()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.to_k_vip = linear_cls(dim, dim, bias=False)
        self.to_v_vip = linear_cls(dim, dim, bias=False)

        self.to_out_vip = nn.ModuleList(
            [
                zero_module(linear_cls(dim, dim, bias=False)),
                nn.Dropout(dropout),
            ]
        )
        
        self.num_vip_tokens = num_vip_tokens
        self.vip_scale = vip_scale

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rope2d = RoPE2D()
            self.rope1d = RoPE1D()
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor_2d = self.rope_scaling["factor_2d"]
            scaling_factor_1d = self.rope_scaling["factor_1d"]
            if scaling_type == "linear":
                self.rope2d = LinearScalingRoPE2D(scaling_factor=scaling_factor_2d)
                self.rope1d = LinearScalingRoPE1D(scaling_factor=scaling_factor_1d)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
            
    def forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        position_q: Optional[torch.LongTensor] = None,
        position_k: Optional[torch.LongTensor] = None,
        last_shape: Tuple[int] = None, 
    ) -> torch.FloatTensor:
        
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)


        if self.compress_kv_factor is not None:
            batch_size = hidden_states.shape[0]
            if len(last_shape) == 2:
                encoder_hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, self.dim, *last_shape)
                encoder_hidden_states = attn.sr(encoder_hidden_states).reshape(batch_size, self.dim, -1).permute(0, 2, 1)
            elif len(last_shape) == 1:
                encoder_hidden_states = hidden_states.permute(0, 2, 1)
                if last_shape[0] % 2 == 1:
                    first_frame_pad = encoder_hidden_states[:, :, :1].repeat((1, 1, attn.kernel_size - 1))
                    encoder_hidden_states = torch.concatenate((first_frame_pad, encoder_hidden_states), dim=2)
                encoder_hidden_states = attn.sr(encoder_hidden_states).permute(0, 2, 1)
            else:
                raise NotImplementedError(f'NotImplementedError with last_shape {last_shape}')
                
            encoder_hidden_states = attn.norm(encoder_hidden_states)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        has_vip_tokens = encoder_hidden_states is not None
        if has_vip_tokens:
            end_pos = sequence_length - self.num_vip_tokens
        
        # attention_mask include encoder_hidden_states(text) and clip_feature(image or video)
        # import ipdb;ipdb.set_trace()
        if attention_mask is not None:
            if has_vip_tokens:
                attention_mask, vip_attention_mask = attention_mask[..., :end_pos], attention_mask[..., end_pos:]
                vip_attention_mask = attn.prepare_attention_mask(vip_attention_mask, self.num_vip_tokens, batch_size)
                vip_attention_mask = vip_attention_mask.view(batch_size, attn.heads, -1, vip_attention_mask.shape[-1])

            attention_mask = attn.prepare_attention_mask(attention_mask, end_pos, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, vip_hidden_states
            if has_vip_tokens:
                encoder_hidden_states, vip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )
            if attn.norm_cross:
                # vip tokens is normed
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        vip_key = self.to_k_vip(vip_hidden_states, *args)
        vip_value = self.to_v_vip(vip_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        vip_key = vip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        vip_value = vip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.use_rope:
            # require the shape of (batch_size x nheads x ntokens x dim)
            if position_q.ndim == 3:
                query = self.rope2d(query, position_q)
            elif position_q.ndim == 2:
                query = self.rope1d(query, position_q)
            else:
                raise NotImplementedError
            if position_k.ndim == 3:
                key = self.rope2d(key, position_k)
                vip_key = self.rope2d(vip_key, position_k)
            elif position_k.ndim == 2:
                key = self.rope1d(key, position_k)
                vip_key = self.rope1d(vip_key, position_k)
            else:
                raise NotImplementedError

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.attention_mode == 'flash':
            assert attention_mask is None or torch.all(attention_mask.bool()), 'flash-attn do not support attention_mask'
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                vip_hidden_states = F.scaled_dot_product_attention(
                    query, vip_key, vip_value, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'xformers':
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                vip_hidden_states = F.scaled_dot_product_attention(
                    query, vip_key, vip_value, attn_mask=vip_attention_mask, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'math':
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            vip_hidden_states = F.scaled_dot_product_attention(
                query, vip_key, vip_value, attn_mask=vip_attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        vip_hidden_states = vip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        vip_hidden_states = vip_hidden_states.to(query.dtype)

        # linear proj
        vip_hidden_states = self.to_out_vip[0](vip_hidden_states, *args)
        # dropout
        vip_hidden_states = self.to_out_vip[1](vip_hidden_states)

        hidden_states = hidden_states + self.vip_scale * vip_hidden_states

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

if __name__ == "__main__":
    model = VideoIPVideoEncoder(
        image_encoder_type="clip",
        inner_dim=1024,
        cross_attention_dim=1152,
        num_attention_layers=2,
        use_rope=False,
        rope_scaling=None,
        attention_mode='math',
        vae_scale_factor_t=4,
        video_length=17,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params / 1e6}M")

    video = torch.randn(2, 1280, 1, 16, 16)

    output = model(video, training_image=True)
    print(output.vip_cond_mask)

