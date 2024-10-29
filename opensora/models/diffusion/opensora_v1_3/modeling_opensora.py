import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.opensora_v1_3.modules import BasicTransformerBlock, Attention
from opensora.models.diffusion.common import PatchEmbed2D
from opensora.utils.utils import to_2tuple
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

class OpenSoraT2V_v1_3(ModelMixin, ConfigMixin):
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
    ):
        super().__init__()
        # Set some common variables used across the board.
        self.out_channels = in_channels if out_channels is None else out_channels
        self.config.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        self._init_patched_inputs()

    def _init_patched_inputs(self):

        self.config.sample_size = (self.config.sample_size_h, self.config.sample_size_w)
        interpolation_scale_thw = (
            self.config.interpolation_scale_t, 
            self.config.interpolation_scale_h, 
            self.config.interpolation_scale_w
            )
        
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels, hidden_size=self.config.hidden_size
        )

        self.pos_embed = PatchEmbed2D(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )
        
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    interpolation_scale_thw=interpolation_scale_thw, 
                    sparse1d=self.config.sparse1d if i > 1 and i < 30 else False, 
                    sparse_n=self.config.sparse_n, 
                    sparse_group=i % 2 == 1, 
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.config.hidden_size) / self.config.hidden_size**0.5)
        self.proj_out = nn.Linear(
            self.config.hidden_size, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
        )
        self.adaln_single = AdaLayerNormSingle(self.config.hidden_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
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
            # b, frame, h, w -> a video
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


        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1, l
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0


        # 1. Input
        frame = ((frame - 1) // self.config.patch_size_t + 1) if frame % 2 == 1 else frame // self.config.patch_size_t  # patchfy
        height, width = hidden_states.shape[-2] // self.config.patch_size, hidden_states.shape[-1] // self.config.patch_size


        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, batch_size, frame
        )

        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

        sparse_mask = {}
        if npu_config is None:
            if get_sequence_parallel_state():
                head_num = self.config.num_attention_heads // nccl_info.world_size
            else:
                head_num = self.config.num_attention_heads
        else:
            head_num = None
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = Attention.prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n, head_num)
        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            if i > 1 and i < 30:
                attention_mask, encoder_attention_mask = sparse_mask[block.attn1.processor.sparse_n][block.attn1.processor.sparse_group]
            else:
                attention_mask, encoder_attention_mask = sparse_mask[1][block.attn1.processor.sparse_group]
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
                    timestep,
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
                    timestep=timestep,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            embedded_timestep=embedded_timestep,
            num_frames=frame, 
            height=height,
            width=width,
        )  # b c t h w

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, batch_size, frame):
        
        hidden_states = self.pos_embed(hidden_states.to(self.dtype))

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d or b, 1, l, d
        assert encoder_hidden_states.shape[1] == 1
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep

    
    
    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, embedded_timestep, num_frames, height, width
    ):  
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, num_frames, height, width, self.config.patch_size_t, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, 
                   num_frames * self.config.patch_size_t, height * self.config.patch_size, width * self.config.patch_size)
        )
        return output

def OpenSoraT2V_v1_3_2B_122(**kwargs):
    kwargs.pop('skip_connection', None)
    return OpenSoraT2V_v1_3(
        num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
        caption_channels=4096, cross_attention_dim=2304, activation_fn="gelu-approximate", **kwargs
        )

OpenSora_v1_3_models = {
    "OpenSoraT2V_v1_3-2B/122": OpenSoraT2V_v1_3_2B_122,  # 2.7B
}

OpenSora_v1_3_models_class = {
    "OpenSoraT2V_v1_3-2B/122": OpenSoraT2V_v1_3,
    "OpenSoraT2V_v1_3": OpenSoraT2V_v1_3,
}

if __name__ == '__main__':
    from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
    from opensora.models.causalvideovae import ae_norm, ae_denorm
    from opensora.models import CausalVAEModelWrapper

    args = type('args', (), 
    {
        'ae': 'WFVAEModel_D8_4x8x8', 
        'model_max_length': 300, 
        'max_height': 176,
        'max_width': 176,
        'num_frames': 33,
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        "sparse1d": True, 
        "sparse_n": 4, 
        "rank": 64, 
    }
    )
    b = 2
    c = 8
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

    device = torch.device('cuda:0')
    model = OpenSoraT2V_v1_3_2B_122(
        in_channels=c, 
        out_channels=c, 
        sample_size_h=latent_size, 
        sample_size_w=latent_size, 
        sample_size_t=num_frames, 
        activation_fn="gelu-approximate",
        attention_bias=True,
        double_self_attention=False,
        norm_elementwise_affine=False,
        norm_eps=1e-06,
        only_cross_attention=False,
        upcast_attention=False,
        interpolation_scale_t=args.interpolation_scale_t, 
        interpolation_scale_h=args.interpolation_scale_h, 
        interpolation_scale_w=args.interpolation_scale_w, 
        sparse1d=args.sparse1d, 
        sparse_n=args.sparse_n
    )
    
    try:
        path = "/storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x1_anyx93x640x640_fps16_lr1e-5_snr5_ema9999_sparse1d4_dit_l_mt5xxl_vpred_zerosnr/checkpoint-43000/model_ema/diffusion_pytorch_model.safetensors"
        # ckpt = torch.load(path, map_location="cpu")
        from safetensors.torch import load_file as safe_load
        ckpt = safe_load(path, device="cpu")
        msg = model.load_state_dict(ckpt, strict=True)
        print(msg)
    except Exception as e:
        print(e)
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    # import sys;sys.exit()
    model = model.to(device)
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1, args.model_max_length)).to(device)  # B L or B 1+num_images L
    timestep = torch.randint(0, 1000, (b,), device=device)
    model_kwargs = dict(
        hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask, 
        encoder_attention_mask=cond_mask, timestep=timestep
        )
    with torch.no_grad():
        output = model(**model_kwargs)
    print(output[0].shape)

