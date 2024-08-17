
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

from .modules import BasicTransformerBlock, get_1d_sincos_pos_embed
from einops import rearrange

from .rope import PositionGetter3D, RoPE3D
try:
    import torch_npu
    from opensora.npu_config import npu_config, set_run_dtype
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
    from opensora.acceleration.communications import all_to_all_SBH
except:
    torch_npu = None
    npu_config = None
    set_run_dtype = None

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
        in_channels=1024,
        num_attention_heads=16,
        attention_head_dim=72, 
        cross_attention_dim=2304,
        num_attention_layers=[1, 3],
        use_rope=False,
        attention_mode='xformers',
        vae_scale_factor_t=4,
        num_frames=93, # when image mode, num_frames = 1; when video mode, num_frames = 93 
        max_num_tokens=288, # when 480p, max_num_tokens = 24 * 3 * 4 = 288; when 720p or 1080p, max_num_tokens = 24 * 4 * 7 = 672
        pooled_token_output_size=(12, 16), # when 480p, size=(12, 16); when 720p or 1080p, size=(16, 28)
        interpolation_scale_thw=(1, 1, 1),
    ):
        super().__init__()

        if USE_PEFT_BACKEND:
            linear_cls = nn.Linear
        else:
            linear_cls = LoRACompatibleLinear

        inner_dim = num_attention_heads * attention_head_dim # 3d rope need inner_dim to be multiple of 3
        assert inner_dim % 3 == 0, "inner_dim must be multiple of 3"

        self.vae_scale_factor_t = vae_scale_factor_t
        self.num_frames = num_frames

        self.max_num_tokens = max_num_tokens

        self.use_rope = use_rope

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=pooled_token_output_size)
        self.proj_in = nn.Sequential( 
            linear_cls(in_channels, inner_dim),
            nn.GELU(),
            linear_cls(inner_dim, inner_dim),
        )

        if not use_rope:
            temp_pos_embed = get_1d_sincos_pos_embed(inner_dim, self.num_frames, base_size=self.num_frames, interpolation_scale=1.0)
            self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

        self.conv1 = nn.Conv3d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, stride=2, padding=1) # F * H * W -> (F // 2) * (H // 2) * (W // 2)

        self.conv2 = nn.Conv3d(in_channels=inner_dim, out_channels=inner_dim, kernel_size=3, stride=2, padding=1) # (F // 2) * (H // 2) * (W // 2) -> (F // 4) * (H // 4) * (W // 4)

        self.trans1= nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=0.0,
                    cross_attention_dim=None,
                    double_self_attention=False,
                    activation_fn="geglu",
                    norm_type="layer_norm",
                    use_rope=use_rope,
                    attention_mode=attention_mode,
                    interpolation_scale_thw=interpolation_scale_thw,
                )
                for _ in range(num_attention_layers[0])
            ]
        ) # only self-attention

        self.trans2 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=0.0,
                    cross_attention_dim=None,
                    double_self_attention=False,
                    activation_fn="geglu",
                    norm_type="layer_norm",
                    use_rope=use_rope,
                    attention_mode=attention_mode,
                    interpolation_scale_thw=interpolation_scale_thw,
                )
                for _ in range(num_attention_layers[1])
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
        
        # when 480p, B C F 37 49 -> B C F 12 16; when 720p or 1080p, B C F 37 65 -> B C F 16 28
        hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w') 
        hidden_states = self.avg_pool(hidden_states) # (B F) C H W -> (B F) C h w
        hidden_states = rearrange(hidden_states, '(b f) c h w -> b f h w c', f=input_frame)
        hidden_states = self.proj_in(hidden_states)

        if not self.use_rope:
            temp_pos_embed = self.temp_pos_embed
            temp_pos_embed = rearrange(temp_pos_embed, 'b f c -> b f 1 1 c')

            hidden_states = hidden_states + temp_pos_embed[:, :input_frame]

        hidden_states = rearrange(hidden_states, 'b f h w c -> b c f h w')

        if image_mode: 
            hidden_states = hidden_states.repeat_interleave(self.vae_scale_factor_t, dim=2)
            hidden_states = rearrange(hidden_states, 'b c (f i) h w -> (b f) c i h w', f=input_frame, i=self.vae_scale_factor_t)
        else:
            image_hidden_states = hidden_states[:, :, 0:1].repeat(1, 1, self.vae_scale_factor_t, 1, 1)
            hidden_states = torch.cat([image_hidden_states, hidden_states[:, :, 1:]], dim=2)
        
        hidden_states = self.conv1(hidden_states) # B C F h w -> B C (F // 2) (h // 2) (w // 2)

        # after conv1
        frame, height, width = hidden_states.shape[2:]
        # if training image, now batch = input_batch_size * frame; if not, batch remains the same
        hidden_states = rearrange(hidden_states, 'b c f h w -> b (f h w) c')


        for layer in self.trans1:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                frame=frame,
                height=height,
                width=width,
            )
        
        # when using image mode, f=1; when using video mode, f=video_length 
        hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w ", f=frame, h=height, w=width)
        
        hidden_states = self.conv2(hidden_states) # B C (F // 2) (h // 2) (w // 2) -> B C (F // 4) (h // 4) (w // 4)

        # after conv2
        frame, height, width = hidden_states.shape[2:]
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")

        for layer in self.trans2:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                frame=frame,
                height=height,
                width=width,
            )

        # when using image mode, n = 1 * h * w; when using video mode, n = video_length * h * w
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
        in_channels=1024,
        num_attention_heads=16,
        attention_head_dim=72,
        cross_attention_dim=2304,
        max_num_tokens=288,
        pooled_token_output_size=(12, 16),
        num_attention_layers=[1, 3],
        use_rope=True,
        attention_mode='math',
        gradient_checkpointing=False,
        vae_scale_factor_t=4,
        num_frames=93,

    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = VideoIPVideoEncoder(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            num_attention_layers=num_attention_layers,
            use_rope=use_rope,
            attention_mode=attention_mode,
            vae_scale_factor_t=vae_scale_factor_t,
            num_frames=num_frames,
            max_num_tokens=max_num_tokens,
            pooled_token_output_size=pooled_token_output_size,
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
        dim=2304, 
        attention_mode='xformers', 

        num_vip_tokens=288,
        vip_scale=1.0,
        dropout=0.0,
    ):
        super().__init__()

        self.attention_mode = attention_mode

        if USE_PEFT_BACKEND:
            linear_cls = nn.Linear
        else:
            linear_cls = LoRACompatibleLinear

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

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        frame: int = 8, 
        height: int = 16, 
        width: int = 16, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if attn.downsampler is not None:
            hidden_states, attention_mask = attn.downsampler(hidden_states, attention_mask, t=frame, h=height, w=width)
            frame, height, width = attn.downsampler.t, attn.downsampler.h, attn.downsampler.w

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)


        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        has_vip_tokens = encoder_hidden_states is not None
        if has_vip_tokens:
            end_pos = sequence_length - self.num_vip_tokens

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

        query = attn.to_q(hidden_states)

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

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        vip_key = self.to_k_vip(vip_hidden_states)
        vip_value = self.to_v_vip(vip_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        vip_key = vip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # qk norm
        # query = attn.q_norm(query)
        # key = attn.k_norm(key)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        vip_value = vip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # if attention_mask is None or not torch.all(attention_mask.bool()):  # 0 mean visible
        #     attention_mask = None
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # import ipdb;ipdb.set_trace()
        # print(attention_mask)
        if self.attention_mode == 'flash':
            assert attention_mask is None or not torch.all(attention_mask.bool()), 'flash-attn do not support attention_mask'
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
        vip_hidden_states = vip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)
        vip_hidden_states = vip_hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # linear proj
        vip_hidden_states = self.to_out_vip[0](vip_hidden_states)
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
        in_channels=1536,
        num_attention_heads=16,
        attention_head_dim=72,
        cross_attention_dim=2304,
        num_attention_layers=[1, 3],
        use_rope=True,
        attention_mode='math',
        vae_scale_factor_t=4,
        num_frames=1,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params / 1e6}M")

    video = torch.randn(2, 1536, 1, 45, 37)

    output = model(video, image_mode=True)

