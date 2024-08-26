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
from opensora.models.diffusion.opensora2.modules import MotionAdaLayerNormSingle, PatchEmbed2D, BasicTransformerBlock
from opensora.utils.utils import to_2tuple
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

class OpenSoraT2V(ModelMixin, ConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

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
    ):
        super().__init__()

        # Validate inputs.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        # Set some common variables used across the board.
        self.use_motion = use_motion
        self.sparse1d = sparse1d
        self.sparse2d = sparse2d
        self.sparse_n = sparse_n
        self.use_rope = use_rope
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale_t = interpolation_scale_t
        self.interpolation_scale_h = interpolation_scale_h
        self.interpolation_scale_w = interpolation_scale_w
        self.downsampler = downsampler
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        self.config.hidden_size = self.inner_dim
        use_additional_conditions = False
        # if use_additional_conditions is None:
            # if norm_type == "ada_norm_single" and sample_size == 128:
            #     use_additional_conditions = True
            # else:
            # use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        assert in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`. Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Initialize the right blocks.
        # Initialize the output blocks and other projection blocks when necessary.
        self._init_patched_inputs(norm_type=norm_type)

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size_t is not None, "OpenSoraT2V over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraT2V over patched input must provide sample_size"
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
        self.pos_embed = PatchEmbed2D(
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
        )
        interpolation_scale_thw = (interpolation_scale_t, *interpolation_scale)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                    attention_mode=self.config.attention_mode, 
                    downsampler=self.config.downsampler, 
                    use_rope=self.config.use_rope, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                    sparse1d=self.sparse1d if i > 1 and i < 30 else False, 
                    sparse2d=self.sparse2d if i > 1 and i < 30 else False, 
                    sparse_n=self.sparse_n, 
                    sparse_group=i % 2 == 1, 
                )
                for i in range(self.config.num_layers)
            ]
        )

        if self.config.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(
                self.inner_dim, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
            )
        elif self.config.norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
            self.proj_out = nn.Linear(
                self.inner_dim, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
            )

        # PixArt-Alpha blocks.
        self.adaln_single = None
        if self.config.norm_type == "ada_norm_single":
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels, hidden_size=self.inner_dim
            )
        self.motion_projection = None
        if self.use_motion:
            self.motion_projection = MotionAdaLayerNormSingle(self.inner_dim)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

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
        motion_score: Optional[torch.FloatTensor] = None,
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
        assert use_image_num == 0
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
                    attention_mask = attention_mask[:, :frame * hccl_info.world_size]  # b, frame, h, w
                else:
                    attention_mask = attention_mask[:, :frame * nccl_info.world_size]  # b, frame, h, w
            else:
                attention_mask = attention_mask[:, :frame]  # b, frame, h, w

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.max_pool3d(attention_mask, kernel_size=(self.patch_size_t, self.patch_size, self.patch_size), 
                                                stride=(self.patch_size_t, self.patch_size, self.patch_size))
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)') 
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0


        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        # import ipdb;ipdb.set_trace()
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0


        if npu_config is not None and attention_mask is not None:
            attention_mask = npu_config.get_attention_mask(attention_mask, attention_mask.shape[-1])
            encoder_attention_mask = npu_config.get_attention_mask(encoder_attention_mask, attention_mask.shape[-2])


        # 1. Input
        frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
        # print('frame', frame)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, motion_score, batch_size, frame, use_image_num
        )
        # 2. Blocks
        # import ipdb;ipdb.set_trace()
        if get_sequence_parallel_state():
            hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
            timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

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
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
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
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )

        if get_sequence_parallel_state():
            hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep,
            num_frames=frame, 
            height=height,
            width=width,
        )  # b c t h w

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, motion_score, batch_size, frame, use_image_num):
        
        hidden_states = self.pos_embed(hidden_states.to(self.dtype), frame)
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
        return output

def OpenSoraT2V_S_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=8, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=768, **kwargs)

def OpenSoraT2V_B_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraT2V_L_122(**kwargs):
    return OpenSoraT2V(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)

OpenSora2_models = {
    "OpenSoraT2V2-S/122": OpenSoraT2V_S_122,  # 0.3B
    "OpenSoraT2V2-B/122": OpenSoraT2V_B_122,  # 1.2B
    "OpenSoraT2V2-L/122": OpenSoraT2V_L_122,  # 2.7B
}

OpenSora2_models_class = {
    "OpenSoraT2V2-S/122": OpenSoraT2V,
    "OpenSoraT2V2-B/122": OpenSoraT2V,
    "OpenSoraT2V2-L/122": OpenSoraT2V,
}

if __name__ == '__main__':
    from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
    from opensora.models.causalvideovae import ae_norm, ae_denorm
    from opensora.models import CausalVAEModelWrapper

    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_D8_4x8x8', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 480,
        'max_width': 640,
        'num_frames': 29,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        "sparse1d": True, 
        "sparse2d": False, 
        "sparse_n": 4, 
        "rank": 64, 
    }
    )
    b = 16
    c = 4
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

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
                            sparse1d=args.sparse1d, 
                            sparse2d=args.sparse2d, 
                            sparse_n=args.sparse_n
                            ).to(device)
    
    try:
        path = "/storage/dataset/Open-Sora-Plan-v1.2.0/29x720p/diffusion_pytorch_model.safetensors"
        ckpt = torch.load(path, map_location="cpu")
        msg = model.load_state_dict(ckpt, strict=True)
        print(msg)
    except Exception as e:
        print(e)
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    # import sys;sys.exit()
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

