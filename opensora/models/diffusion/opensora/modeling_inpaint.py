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
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

from PIL import Image
import numpy as np
from enum import Enum, auto
import glob

from .modeling_opensora import OpenSoraT2V
from .videoip import VideoIPAdapter, VideoIPAttnProcessor

class ModelType(Enum):
    VIP_ONLY = auto()
    INPAINT_ONLY = auto()
    VIP_INPAINT = auto()

TYPE_TO_STR = {
    ModelType.VIP_ONLY: "vip_only",
    ModelType.INPAINT_ONLY: "inpaint_only",
    ModelType.VIP_INPAINT: "vip_inpaint",
}

STR_TO_TYPE = {v: k for k, v in TYPE_TO_STR.items()}

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

class VIPNet(nn.Module):
    def __init__(
        self,
        image_encoder_out_channels=1536,
        cross_attention_dim=2304,
        num_tokens=16, # when 480p, max_num_tokens = 24 * 3 * 4 = 288; when 720p or 1080p, max_num_tokens = 24 * 4 * 7 = 672
        pooled_token_output_size=(12, 16),
        vip_num_attention_heads=16,
        vip_attention_head_dim=72,
        vip_num_attention_layers=[1, 3],
        attention_mode='xformers',
        gradient_checkpointing=False,
        vae_scale_factor_t=4,
        num_frames=93,
        use_rope=True,
        attn_proc_names=None,
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.attention_mode = attention_mode
        self.use_rope = use_rope

        self.vip_adapter = VideoIPAdapter(
            in_channels=image_encoder_out_channels,
            num_attention_heads=vip_num_attention_heads,
            attention_head_dim=vip_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            max_num_tokens=num_tokens,
            pooled_token_output_size=pooled_token_output_size,
            num_attention_layers=vip_num_attention_layers,
            attention_mode=attention_mode,
            gradient_checkpointing=gradient_checkpointing,
            vae_scale_factor_t=vae_scale_factor_t,
            num_frames=num_frames,
            use_rope=use_rope,
        )

        self.attn_procs = {}
        for name in attn_proc_names:
            if name.endswith('.attn2.processor'):
                self.attn_procs[name] = VideoIPAttnProcessor(
                    dim=cross_attention_dim,
                    attention_mode=attention_mode,
                    num_vip_tokens=num_tokens,
                )
    
    def custom_requires_grad(self, requires_grad=True):
        self.vip_adapter.requires_grad_(requires_grad)
        for module in self.attn_procs.values():
            module.requires_grad_(requires_grad)
        
    def set_vip_adapter(self, model, init_from_original_attn_processor):
        # init adapter modules
        model_sd = model.state_dict()
        attn_procs = {}
        print("set vip adapter...")
        for name, attn_processor in model.attn_processors.items():
            if name.endswith(".attn2.processor"):
                new_attn_processor = self.attn_procs[name]
                if init_from_original_attn_processor: 
                    print(f"init from original attn processor {name}...")
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_vip.weight": model_sd[layer_name + ".to_k.weight"],
                        "to_v_vip.weight": model_sd[layer_name + ".to_v.weight"],
                    }
                    new_attn_processor.load_state_dict(weights, strict=False)
                attn_procs[name] = new_attn_processor
            else:
                attn_procs[name] = attn_processor
                
        model.set_attn_processor(attn_procs)

    def custom_load_state_dict(self, pretrained_model_path):

        print(f'Loading {self.__class__.__name__} pretrained weights...')
        print(f'Loading pretrained model from {pretrained_model_path}...')

        model_state_num = 0
        vip_adapter_state_dict = self.vip_adapter.state_dict()
        model_state_num += len(vip_adapter_state_dict)
        attn_procs_state_dict = {}
        for k, module in self.attn_procs.items():
            attn_procs_state_dict[k] = module.state_dict()
            model_state_num += len(attn_procs_state_dict[k])

        if 'safetensors' in pretrained_model_path:  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(pretrained_model_path, device="cpu")
        else:  # latest stage training weight
            pretrained_checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model' in pretrained_checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']

        pretrained_vip_adapter_checkpoint = pretrained_checkpoint['vip_adapter']
        pretrained_attn_procs_checkpoint = pretrained_checkpoint['attn_procs']

        vip_adapter_checkpoint = reconstitute_checkpoint(pretrained_vip_adapter_checkpoint, vip_adapter_state_dict)
        attn_procs_checkpoints = {}
        for name, checkpoint in pretrained_attn_procs_checkpoint.items():
            attn_procs_checkpoints[name] = reconstitute_checkpoint(checkpoint, attn_procs_state_dict[name])
        
        missing_keys, unexpected_keys = self.vip_adapter.load_state_dict(vip_adapter_checkpoint, strict=False)

        for name, checkpoint in attn_procs_checkpoints.items():
            m, u = self.attn_procs[name].load_state_dict(checkpoint, strict=False)
            missing_keys += m
            unexpected_keys += u

        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {model_state_num - len(missing_keys)}/{model_state_num} keys from {pretrained_model_path}!')
            
    def register_get_clip_features_func(self, get_clip_features):
        self.get_clip_features = get_clip_features

    @torch.no_grad()
    def get_image_embeds(self, images, image_processor, image_encoder, transform, device, weight_dtype=torch.float32):
        if not isinstance(images, list):
            images = [images]
        images = [Image.open(image).convert("RGB") for image in images]
        images = [torch.from_numpy(np.copy(np.array(image))).unsqueeze(0) for image in images] # 1 H W C
        images = torch.cat([transform(image.permute(0, 3, 1, 2)) for image in images]) # resize, 1 C H W

        images = image_processor(images) # 1 C H W
        images = images.to(device=device, dtype=weight_dtype)
        negative_images = torch.zeros_like(images, device=device, dtype=weight_dtype)

        images = images.unsqueeze(0) # 1 1 C H W
        negative_images = negative_images.unsqueeze(0)

        clip_features = self.get_clip_features(images, image_encoder) # 1 1 C H W -> 1 D 1 h w
        negative_clip_features = self.get_clip_features(negative_images, image_encoder)

        return clip_features, negative_clip_features

    @torch.no_grad()
    def get_video_embeds(self, condition_images, num_frames, image_processor, image_encoder, transform, device, weight_dtype=torch.float32):
        if len(condition_images) == 1:
            condition_images_indices = [0]
        elif len(condition_images) == 2:
            condition_images_indices = [0, -1]
        condition_images = [Image.open(image).convert("RGB") for image in condition_images]
        condition_images = [torch.from_numpy(np.copy(np.array(image))).unsqueeze(0) for image in condition_images] # F [1 H W C]
        condition_images = torch.cat([transform(image.permute(0, 3, 1, 2)) for image in condition_images]) # resize, [F C H W]

        condition_images = image_processor(condition_images) # F C H W
        condition_images = condition_images.to(device=device, dtype=weight_dtype)
        _, C, H, W = condition_images.shape
        video = torch.zeros([num_frames, C, H, W], device=device, dtype=weight_dtype)
        video[condition_images_indices] = condition_images
        negative_video = torch.zeros_like(video, device=device, dtype=weight_dtype)

        video = video.unsqueeze(0) # 1 F C H W
        negative_video = negative_video.unsqueeze(0)

        # using the second last layer of image encoder
        clip_features = self.get_clip_features(video, image_encoder) # 1 F C H W -> 1 D F h w
        negative_clip_features = self.get_clip_features(negative_video, image_encoder)

        return clip_features, negative_clip_features


    def forward(self, clip_feature, use_image_num):

        vip_out = self.vip_adapter(hidden_states=clip_feature, use_image_num=use_image_num) # B D T H W  -> B 1 N D
        vip_tokens, vip_cond_mask = vip_out.hidden_states, vip_out.vip_cond_mask

        return dict(vip_tokens=vip_tokens, vip_cond_mask=vip_cond_mask)
    

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

        model_type = 'inpaint_only',
        vae_scale_factor_t=4,
        image_encoder_out_channels=1536,
        vip_num_attention_heads=16,
        vip_attention_head_dim=72,
        vip_num_attention_layers=[1, 3],
        vip_gradient_checkpointing=False,
        num_frames=93,
        vip_init_from_original_attn_processor=False,
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

        self.model_type = STR_TO_TYPE[model_type]

        if self.model_type != ModelType.VIP_ONLY:
            self.vae_scale_factor_t = vae_scale_factor_t
            # init masked_video and mask conv_in
            self._init_patched_inputs_for_inpainting()

        if self.model_type != ModelType.INPAINT_ONLY:
            # init vipnet
            self._init_vip_net(
                image_encoder_out_channels=image_encoder_out_channels,
                cross_attention_dim=cross_attention_dim,
                vip_num_attention_heads=vip_num_attention_heads,
                vip_attention_head_dim=vip_attention_head_dim,
                vip_num_attention_layers=vip_num_attention_layers,
                attention_mode=attention_mode,
                gradient_checkpointing=vip_gradient_checkpointing,
                vae_scale_factor_t=vae_scale_factor_t,
                num_frames=num_frames,
                use_rope=use_rope,
            )
            
            self.vip.set_vip_adapter(self, init_from_original_attn_processor=vip_init_from_original_attn_processor)


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
        
        if self.config.downsampler is not None and len(self.config.downsampler) == 9:
            self.pos_embed_mask = nn.ModuleList(
                [
                    OverlapPatchEmbed3D(
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
            self.pos_embed_masked_video = nn.ModuleList(
                [
                    OverlapPatchEmbed3D(
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
        elif self.config.downsampler is not None and len(self.config.downsampler) == 7:
            self.pos_embed_mask = nn.ModuleList(
                [
                    OverlapPatchEmbed2D(
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
            
            self.pos_embed_masked_video = nn.ModuleList(
                [
                    OverlapPatchEmbed2D(
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
        
        else:
            self.pos_embed_mask = nn.ModuleList(
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
            self.pos_embed_masked_video = nn.ModuleList(
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

    def _init_vip_net(
        self,
        image_encoder_out_channels=1536,
        cross_attention_dim=2304,
        vip_num_attention_heads=16,
        vip_attention_head_dim=72,
        vip_num_attention_layers=[1, 3],
        attention_mode='xformers',
        gradient_checkpointing=False,
        vae_scale_factor_t=4,
        num_frames=93,
        use_rope=True,
        init_from_original_attn_processor=False,
    ):
        attn_proc_names = self.attn_processors.keys()

        if self.config.sample_size[0] / self.config.sample_size[1] == 9 / 16:
            pooled_token_output_size = (16, 28) # 720p or 1080p
        elif self.config.sample_size[0] / self.config.sample_size[1] == 3 / 4:
            pooled_token_output_size = (12, 16) # 480p
        else:
            raise NotImplementedError
        
        # when 480p, max_num_tokens = 24 * 3 * 4 = 288; when 720p or 1080p, max_num_tokens = 24 * 4 * 7 = 672
        num_tokens = pooled_token_output_size[0] // 4 * pooled_token_output_size[1] // 4 * self.config.sample_size_t

        print(f"initialize VIPNet, num_tokens: {num_tokens}, pooled_token_output_size: {pooled_token_output_size}")

        self.vip = VIPNet(
            image_encoder_out_channels=image_encoder_out_channels,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens, 
            pooled_token_output_size=pooled_token_output_size,
            vip_num_attention_heads=vip_num_attention_heads,
            vip_attention_head_dim=vip_attention_head_dim,
            vip_num_attention_layers=vip_num_attention_layers,
            attention_mode=attention_mode,
            gradient_checkpointing=gradient_checkpointing,
            vae_scale_factor_t=vae_scale_factor_t,
            num_frames=num_frames,
            use_rope=use_rope,
            attn_proc_names=attn_proc_names,
        )

        self.vip.set_vip_adapter(self, init_from_original_attn_processor=init_from_original_attn_processor)

        
    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num, vip_tokens=None):
        if self.model_type != ModelType.VIP_ONLY:
            # inpaint
            assert hidden_states.shape[1] == 2 * self.config.in_channels + self.vae_scale_factor_t
            in_channels = self.config.in_channels
            hidden_states, hidden_states_masked_vid, hidden_states_mask = hidden_states[:, :in_channels], hidden_states[:, in_channels: 2 * in_channels], hidden_states[:, 2 * in_channels:]

            hidden_states_vid, hidden_states_img = self.pos_embed(hidden_states.to(self.dtype), frame)
            hidden_states_masked_vid, _ = self.pos_embed_masked_video[0](hidden_states_masked_vid.to(self.dtype), frame)
            hidden_states_masked_vid = self.pos_embed_masked_video[1](hidden_states_masked_vid)

            hidden_states_mask, _ = self.pos_embed_mask[0](hidden_states_mask.to(self.dtype), frame)
            hidden_states_mask = self.pos_embed_mask[1](hidden_states_mask)

            hidden_states_vid = hidden_states_vid + hidden_states_masked_vid + hidden_states_mask
        else:
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

            if self.model_type != ModelType.INPAINT_ONLY:
                # NOTE add vip hidden states
                encoder_hidden_states = torch.cat([encoder_hidden_states, vip_tokens], dim=2)  # # B 1 N D -> B 1 N+num_vip_tokens D

            if hidden_states_vid is None:
                encoder_hidden_states_img = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')
            else:
                encoder_hidden_states_vid = rearrange(encoder_hidden_states[:, :1], 'b 1 l d -> (b 1) l d')
                if hidden_states_img is not None:
                    encoder_hidden_states_img = rearrange(encoder_hidden_states[:, 1:], 'b i l d -> (b i) l d')


        return hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img
    
    def transformer_model_custom_load_state_dict(self, pretrained_model_path, load_mask=False, model_type=ModelType.INPAINT_ONLY):
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

        if model_type != ModelType.VIP_ONLY:
            if not 'pos_embed_masked_video.0.weight' in checkpoint:
                checkpoint['pos_embed_masked_video.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
                checkpoint['pos_embed_masked_video.0.proj.bias'] = checkpoint['pos_embed.proj.bias']
            if not 'pos_embed_mask.0.proj.weight' in checkpoint and load_mask:
                checkpoint['pos_embed_mask.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
                checkpoint['pos_embed_mask.0.proj.bias'] = checkpoint['pos_embed.proj.bias']

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {len(self.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {pretrained_model_path}!')

    def custom_load_state_dict(self, pretrained_model_path, load_mask=False):
        assert isinstance(pretrained_model_path, dict), "pretrained_model_path must be a dict"

        pretrained_transformer_model_path = pretrained_model_path.get('transformer_model', None)
        pretrained_vip_path = pretrained_model_path.get('vip', None)

        self.transformer_model_custom_load_state_dict(pretrained_transformer_model_path, load_mask, model_type=self.model_type)
        if self.model_type != ModelType.INPAINT_ONLY and pretrained_vip_path is not None:
            self.vip.custom_load_state_dict(pretrained_vip_path)

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
        clip_features: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        return_dict: bool = True,
    ):
        
        batch_size, c, frame, h, w = hidden_states.shape
        # print('hidden_states.shape', hidden_states.shape)
        frame = frame - use_image_num  # 21-4=17
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # video ip adapter forward
        if self.model_type != ModelType.INPAINT_ONLY:
            vip_out = self.vip(clip_features, use_image_num=use_image_num)
            vip_tokens, vip_attention_mask = vip_out['vip_tokens'], vip_out['vip_cond_mask']

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
            if get_sequence_parallel_state():
                if npu_config is not None:
                    attention_mask_vid = attention_mask[:, :frame * hccl_info.world_size]  # b, frame, h, w
                    attention_mask_img = attention_mask[:, frame * hccl_info.world_size:]  # b, use_image_num, h, w
                else:
                    # print('before attention_mask.shape', attention_mask.shape)
                    attention_mask_vid = attention_mask[:, :frame * nccl_info.world_size]  # b, frame, h, w
                    attention_mask_img = attention_mask[:, frame * nccl_info.world_size:]  # b, use_image_num, h, w
                    # print('after attention_mask.shape', attention_mask_vid.shape)
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

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                attention_mask_img = attention_mask_vid
                attention_mask_vid = None
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        # import ipdb;ipdb.set_trace()
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1+use_image_num, l -> a video with images
            # b, 1, l -> only images
            if self.model_type != ModelType.INPAINT_ONLY:
                # NOTE add vip attention mask
                encoder_attention_mask = torch.cat([encoder_attention_mask, vip_attention_mask], dim=-1) # B 1 N -> B 1 N+num_vip_tokens

            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            in_t = encoder_attention_mask.shape[1]
            encoder_attention_mask_vid = encoder_attention_mask[:, :in_t-use_image_num]  # b, 1, l
            encoder_attention_mask_vid = rearrange(encoder_attention_mask_vid, 'b 1 l -> (b 1) 1 l') if encoder_attention_mask_vid.numel() > 0 else None

            encoder_attention_mask_img = encoder_attention_mask[:, in_t-use_image_num:]  # b, use_image_num, l
            encoder_attention_mask_img = rearrange(encoder_attention_mask_img, 'b i l -> (b i) 1 l') if encoder_attention_mask_img.numel() > 0 else None

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
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
        # print('frame', frame)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.model_type != ModelType.INPAINT_ONLY:
            hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, \
            timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num, vip_tokens=vip_tokens,
            )
        else:
            hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, \
            timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num,
            )
        # 2. Blocks
        # import ipdb;ipdb.set_trace()
        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                # print(333333333333333)
                hidden_states_vid = rearrange(hidden_states_vid, 'b s h -> s b h', b=batch_size).contiguous()
                encoder_hidden_states_vid = rearrange(encoder_hidden_states_vid, 'b s h -> s b h',
                                                        b=batch_size).contiguous()
                timestep_vid = timestep_vid.view(batch_size, 6, -1).transpose(0, 1).contiguous()
                # print('timestep_vid', timestep_vid.shape)

                
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

        if get_sequence_parallel_state():
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


def OpenSoraInpaint_S_111(**kwargs):
    return OpenSoraInpaint(num_layers=28, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=1,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraInpaint_B_111(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=1,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

def OpenSoraInpaint_L_111(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=1,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)

def OpenSoraInpaint_S_122(**kwargs):
    return OpenSoraInpaint(num_layers=28, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraInpaint_B_122(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

def OpenSoraInpaint_L_122(**kwargs):
    return OpenSoraInpaint(num_layers=40, attention_head_dim=128, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2048, **kwargs)

def OpenSoraInpaint_ROPE_L_122(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)
# def OpenSoraInpaint_S_222(**kwargs):
#     return OpenSoraInpaint(num_layers=28, attention_head_dim=72, num_attention_heads=16, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)


def OpenSoraInpaint_B_222(**kwargs):
    return OpenSoraInpaint(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=2, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

# def OpenSoraInpaint_L_222(**kwargs):
#     return OpenSoraInpaint(num_layers=40, attention_head_dim=128, num_attention_heads=20, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2560, **kwargs)

# def OpenSoraInpaint_XL_222(**kwargs):
#     return OpenSoraInpaint(num_layers=32, attention_head_dim=128, num_attention_heads=32, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=4096, **kwargs)

# def OpenSoraInpaint_XXL_222(**kwargs):
#     return OpenSoraInpaint(num_layers=40, attention_head_dim=128, num_attention_heads=40, patch_size_t=2, patch_size=2,
#                        norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=5120, **kwargs)

Inpaint_models = {
    "OpenSoraInpaint-S/122": OpenSoraInpaint_S_122,  #       1.1B
    "OpenSoraInpaint-B/122": OpenSoraInpaint_B_122,
    "OpenSoraInpaint-L/122": OpenSoraInpaint_L_122,
    "OpenSoraInpaint-ROPE-L/122": OpenSoraInpaint_ROPE_L_122,
    "OpenSoraInpaint-S/111": OpenSoraInpaint_S_111,
    "OpenSoraInpaint-B/111": OpenSoraInpaint_B_111,
    "OpenSoraInpaint-L/111": OpenSoraInpaint_L_111,
    # "OpenSoraInpaint-XL/122": OpenSoraInpaint_XL_122,
    # "OpenSoraInpaint-XXL/122": OpenSoraInpaint_XXL_122,
    # "OpenSoraInpaint-S/222": OpenSoraInpaint_S_222,
    "OpenSoraInpaint-B/222": OpenSoraInpaint_B_222,
    # "OpenSoraInpaint-L/222": OpenSoraInpaint_L_222,
    # "OpenSoraInpaint-XL/222": OpenSoraInpaint_XL_222,
    # "OpenSoraInpaint-XXL/222": OpenSoraInpaint_XXL_222,
}

Inpaint_models_class = {
    "OpenSoraInpaint-S/122": OpenSoraInpaint,
    "OpenSoraInpaint-B/122": OpenSoraInpaint,
    "OpenSoraInpaint-L/122": OpenSoraInpaint,
    "OpenSoraInpaint-ROPE-L/122": OpenSoraInpaint,
    "OpenSoraInpaint-S/111": OpenSoraInpaint,
    "OpenSoraInpaint-B/111": OpenSoraInpaint,
    "OpenSoraInpaint-L/111": OpenSoraInpaint,

    "OpenSoraInpaint-B/222": OpenSoraInpaint,
}





