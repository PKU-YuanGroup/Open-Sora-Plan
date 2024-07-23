import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.udit.modules import Upsample2d, Downsample2d, PatchEmbed2D, BasicTransformerBlock, \
    FP32_GELU, FP32_SiLU, FP32_Layernorm
from opensora.utils.utils import to_2tuple
import math
import re
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None

class UDiTT2V(ModelMixin, ConfigMixin, PeftAdapterMixin):
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
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        mlp_ratio: int = 4, 
        depth: Optional[list] = [2, 5, 8, 5, 2],
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
        downsampler: str = 'k333_s222', 
        use_rope: bool = False, 
        use_stable_fp32: bool = False, 
    ):
        super().__init__()

        # Set some common variables used across the board.
        self.use_stable_fp32 = use_stable_fp32
        self.use_rope = use_rope
        self.downsampler = downsampler
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale_t = interpolation_scale_t
        self.interpolation_scale_h = interpolation_scale_h
        self.interpolation_scale_w = interpolation_scale_w
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        use_additional_conditions = False
        # if use_additional_conditions is None:
            # if norm_type == "ada_norm_single" and sample_size == 128:
            #     use_additional_conditions = True
            # else:
            # use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        assert in_channels is not None

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
        if self.use_stable_fp32:
            self._replace_fp32_layers()

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size_t is not None, "OpenSoraT2V over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraT2V over patched input must provide sample_size"

        self.config.sample_size = to_2tuple(self.config.sample_size)
        # self.num_frames = self.config.sample_size_t
        # self.height = self.config.sample_size[0]
        # self.width = self.config.sample_size[1]
        interpolation_scale_t = ((self.config.sample_size_t - 1) // 16 + 1) if self.config.sample_size_t % 2 == 1 else self.config.sample_size_t / 16
        interpolation_scale_t = (
            self.config.interpolation_scale_t if self.config.interpolation_scale_t is not None else interpolation_scale_t
        )
        interpolation_scale = (
            self.config.interpolation_scale_h if self.config.interpolation_scale_h is not None else self.config.sample_size[0] / 30, 
            self.config.interpolation_scale_w if self.config.interpolation_scale_w is not None else self.config.sample_size[1] / 40, 
        )

        # down_factor = list(re.search(r's(\d{2,3})', self.downsampler).group(1))
        # down_factor = [int(i) for i in down_factor]
        # down_factor = down_factor if isinstance(self.config.down_factor, list) else [self.config.down_factor] * 5
        # down_factor = [2] * len(self.config.depth)
        is_video_model = False
        # if self.config.downsampler is not None and len(self.config.downsampler) == 9:
        #     is_video_model = True  # to init weight from image
            # self.pos_embed = OverlapPatchEmbed3D(
            #     num_frames=self.config.sample_size_t,
            #     height=self.config.sample_size[0],
            #     width=self.config.sample_size[1],
            #     patch_size_t=self.config.patch_size_t,
            #     patch_size=self.config.patch_size,
            #     in_channels=self.in_channels,
            #     embed_dim=self.inner_dim,
            #     interpolation_scale=interpolation_scale, 
            #     interpolation_scale_t=interpolation_scale_t,
            #     use_abs_pos=not self.config.use_rope, 
            # )
        # elif self.config.downsampler is not None and len(self.config.downsampler) == 7:
        #     is_video_model = False
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
        # layer_thw = [[self.config.sample_size_t//self.config.patch_size_t, 
        #               (self.config.sample_size[0] + self.config.sample_size[0] % (self.config.patch_size*2))//self.config.patch_size, 
        #               (self.config.sample_size[1] + self.config.sample_size[1] % (self.config.patch_size*2))//self.config.patch_size]]
        interpolation_scale_thw = (interpolation_scale_t, *interpolation_scale)
        # for i in range((len(self.config.depth)-1)//2):
        #     t = layer_thw[i][0] // 2 if layer_thw[i][0] != 1 else 1
        #     h = (layer_thw[i][1] + layer_thw[i][1] % 4) // 2  # why mod 4, because downsample and downsampler in attention
        #     w = (layer_thw[i][2] + layer_thw[i][2] % 4) // 2
        #     layer_thw.append([t, h, w])
        # self.layer_thw = layer_thw
        self.encoder_level_1 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    downsampler=self.config.downsampler, 
                    mlp_ratio=self.config.mlp_ratio, 
                    dropout=self.config.dropout,
                    cross_attention_dim=self.inner_dim,
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
                    use_rope=self.config.use_rope, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )
                for _ in range(self.config.depth[0])
            ]
        )
        # self.down1_2 = Downsample3d(self.inner_dim) if is_video_model else Downsample2d(self.inner_dim)
        self.down1_2 = Downsample2d(self.inner_dim)
        
        self.encoder_level_2 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim * 2,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim * 2,
                    downsampler=self.config.downsampler, 
                    mlp_ratio=self.config.mlp_ratio, 
                    dropout=self.config.dropout,
                    cross_attention_dim=self.inner_dim * 2,
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
                    use_rope=self.config.use_rope, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )
                for _ in range(self.config.depth[1])
            ]
        )
        # self.down2_3 = Downsample3d(self.inner_dim * 2) if is_video_model else Downsample2d(self.inner_dim * 2)
        self.down2_3 = Downsample2d(self.inner_dim * 2)

        self.latent = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim * 4,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim * 4,
                    downsampler=self.config.downsampler, 
                    mlp_ratio=self.config.mlp_ratio, 
                    dropout=self.config.dropout,
                    cross_attention_dim=self.inner_dim * 4,
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
                    use_rope=self.config.use_rope, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )
                for _ in range(self.config.depth[2])
            ]
        )

        # self.up3_2 = Upsample3d(int(self.inner_dim * 4)) if is_video_model else Upsample2d(self.inner_dim * 4)  ## From Level 4 to Level 3
        self.up3_2 = Upsample2d(self.inner_dim * 4)  ## From Level 4 to Level 3
        
        self.reduce_chan_level2_norm = nn.LayerNorm(int(self.inner_dim * 4), elementwise_affine=True, eps=1e-6)
        self.reduce_chan_level2 = nn.Linear(int(self.inner_dim * 4), int(self.inner_dim * 2), bias=True)
        self.decoder_level_2 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim * 2,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim * 2,
                    downsampler=self.config.downsampler, 
                    mlp_ratio=self.config.mlp_ratio, 
                    dropout=self.config.dropout,
                    cross_attention_dim=self.inner_dim * 2,
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
                    use_rope=self.config.use_rope, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )
                for _ in range(self.config.depth[3])
            ]
        )

        # self.up2_1 = Upsample3d(int(self.inner_dim * 2)) if is_video_model else Upsample2d(self.inner_dim * 2)  ## From Level 4 to Level 3
        self.up2_1 = Upsample2d(self.inner_dim * 2)  ## From Level 4 to Level 3
        
        self.reduce_chan_level1_norm = nn.LayerNorm(int(self.inner_dim * 2), elementwise_affine=True, eps=1e-6)
        self.reduce_chan_level1 = nn.Linear(int(self.inner_dim * 2), int(self.inner_dim * 1), bias=True)
        self.decoder_level_1 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    downsampler=self.config.downsampler, 
                    mlp_ratio=self.config.mlp_ratio, 
                    dropout=self.config.dropout,
                    cross_attention_dim=self.inner_dim,
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
                    use_rope=self.config.use_rope, 
                    interpolation_scale_thw=interpolation_scale_thw, 
                )
                for _ in range(self.config.depth[4])
            ]
        )

        if self.config.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(2 * self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(2 * self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(
                2 * self.inner_dim, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
            )
        elif self.config.norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / (self.inner_dim)**0.5)
            self.proj_out = nn.Linear(
                self.inner_dim, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
            )

        # PixArt-Alpha blocks.
        # self.adaln_single = None
        # if self.config.norm_type == "ada_norm_single":
        # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
        # additional conditions until we find better name
        self.adaln_single_1 = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )
        self.adaln_single_2 = AdaLayerNormSingle(
            self.inner_dim * 2, use_additional_conditions=self.use_additional_conditions
        )
        self.adaln_single_3 = AdaLayerNormSingle(
            self.inner_dim * 4, use_additional_conditions=self.use_additional_conditions
        )

        # self.caption_projection = None
        # if self.caption_channels is not None:
        self.caption_projection_1 = PixArtAlphaTextProjection(
            in_features=self.caption_channels, hidden_size=self.inner_dim
        )
        self.caption_projection_2 = PixArtAlphaTextProjection(
            in_features=self.caption_channels, hidden_size=self.inner_dim * 2
        )
        self.caption_projection_3 = PixArtAlphaTextProjection(
            in_features=self.caption_channels, hidden_size=self.inner_dim * 4
        )

    def _replace_fp32_layers(self, module=None):
        if module is None:
            module = self
        for name, submodule in module.named_children():
            if isinstance(submodule, nn.LayerNorm):
                # print(f"Replacing LayerNorm in {name}")
                new_layer = FP32_Layernorm(submodule.normalized_shape, submodule.eps, submodule.elementwise_affine)
                if submodule.elementwise_affine:
                    new_layer.weight.data.copy_(submodule.weight.data.float())
                    if submodule.bias is not None:
                        new_layer.bias.data.copy_(submodule.bias.data.float()) 
                setattr(module, name, new_layer)
            elif isinstance(submodule, nn.SiLU):
                # print(f"Replacing SiLU in {name}")
                setattr(module, name, FP32_SiLU(submodule.inplace))
            elif isinstance(submodule, nn.GELU):
                # print(f"Replacing GELU in {name}")
                setattr(module, name, FP32_GELU(submodule.approximate))
            else:
                self._replace_fp32_layers(submodule) 

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
        batch_size, c, frame, height, width = hidden_states.shape
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
            pad_h_0, pad_w_0 = height % (self.config.patch_size * 2), width % (self.config.patch_size * 2)

            hidden_states = F.pad(hidden_states, (0, pad_w_0, 0, pad_h_0, 0, 0), mode='reflect')
            attention_mask = attention_mask.to(self.dtype)
            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.pad(attention_mask, (0, pad_w_0, 0, pad_h_0, 0, 0))
            attention_mask = F.max_pool3d(attention_mask, kernel_size=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size), 
                                                  stride=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size))
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> b 1 (t h w)') 

            attention_bias = (1 - attention_mask.bool().to(self.dtype)) * -10000.0

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1, l -> only video
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0


        # 1. Input
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        hidden_states, encoder_hidden_states_1, encoder_hidden_states_2, encoder_hidden_states_3, \
            timestep_1, timestep_2, timestep_3, \
                embedded_timestep_1, embedded_timestep_2, embedded_timestep_3 = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
        )
        frame, height, width = frame // self.config.patch_size_t, \
            (height + pad_h_0) // (self.config.patch_size), (width + pad_w_0) // (self.config.patch_size)


        assert not torch.any(torch.isnan(hidden_states)), 'after _operate_on_patched_inputs'
        def create_custom_forward(module, return_dict=None):
            def custom_forward(*inputs):
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)

            return custom_forward
        
        # encoder_1
        out_enc_level1 = hidden_states
        # import ipdb;ipdb.set_trace()
        if self.training and self.gradient_checkpointing:

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            for block in self.encoder_level_1:
                out_enc_level1 = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    out_enc_level1,
                    attention_bias,
                    encoder_hidden_states_1,
                    encoder_attention_mask,
                    timestep_1,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
        else:
            for block in self.encoder_level_1:
                out_enc_level1 = block(
                    out_enc_level1,
                    attention_mask=attention_bias,
                    encoder_hidden_states=encoder_hidden_states_1,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep_1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )
        pad_h_1, pad_w_1 = height % 4, width % 4
        
        inp_enc_level2, attention_bias, attention_mask = self.down1_2(out_enc_level1, attention_mask, frame, height, width, pad_h=pad_h_1, pad_w=pad_w_1)
        # frame, height, width = frame // 2 if frame != 1 else frame, (height + pad_h_1) // 2, (width + pad_w_1) // 2
        height, width = (height + pad_h_1) // 2, (width + pad_w_1) // 2

        # encoder_2
        out_enc_level2 = inp_enc_level2

        if self.training and self.gradient_checkpointing:

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            
            for block in self.encoder_level_2:
                out_enc_level2 = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    out_enc_level2,
                    attention_bias,
                    encoder_hidden_states_2,
                    encoder_attention_mask,
                    timestep_2,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
        else:
            for block in self.encoder_level_2:
                out_enc_level2 = block(
                    out_enc_level2,
                    attention_mask=attention_bias,
                    encoder_hidden_states=encoder_hidden_states_2,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep_2,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )
        pad_h_2, pad_w_2 = height % 4, width % 4
        
        # import ipdb;ipdb.set_trace()
        inp_enc_level3, attention_bias, attention_mask = self.down2_3(out_enc_level2, attention_mask, frame, height, width, pad_h=pad_h_2, pad_w=pad_w_2)
        # frame, height, width = frame // 2 if frame != 1 else frame, (height + pad_h_2) // 2, (width + pad_w_2) // 2
        height, width = (height + pad_h_2) // 2, (width + pad_w_2) // 2

        # latent
        latent = inp_enc_level3
        if self.training and self.gradient_checkpointing:

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            
            for block in self.latent:
                latent = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    latent,
                    attention_bias,
                    encoder_hidden_states_3,
                    encoder_attention_mask,
                    timestep_3,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
        else:
            for block in self.latent:
                latent = block(
                    latent,
                    attention_mask=attention_bias,
                    encoder_hidden_states=encoder_hidden_states_3,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep_3,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )

        # decoder_2
        
        # import ipdb;ipdb.set_trace()
        inp_dec_level2, attention_bias, attention_mask = self.up3_2(latent, attention_mask, frame, height, width, pad_h=pad_h_2, pad_w=pad_w_2)
        # frame, height, width = frame * 2 if frame != 1 else frame, height * 2 - pad_h_2, width * 2 - pad_w_2
        height, width = height * 2 - pad_h_2, width * 2 - pad_w_2
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = self.reduce_chan_level2_norm(inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2

        if self.training and self.gradient_checkpointing:

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            
            for block in self.decoder_level_2:
                out_dec_level2 = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    out_dec_level2,
                    attention_bias,
                    encoder_hidden_states_2,
                    encoder_attention_mask,
                    timestep_2,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
        else:
            for block in self.decoder_level_2:
                out_dec_level2 = block(
                    out_dec_level2,
                    attention_mask=attention_bias,
                    encoder_hidden_states=encoder_hidden_states_2,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep_2,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )

        # decoder_1
        
        # import ipdb;ipdb.set_trace()
        inp_dec_level1, attention_bias, attention_mask = self.up2_1(out_dec_level2, attention_mask, frame, height, width, pad_h=pad_h_1, pad_w=pad_w_1)
        # frame, height, width = frame * 2 if frame != 1 else frame, height * 2 - pad_h_1, width * 2 - pad_w_1
        height, width = height * 2 - pad_h_1, width * 2 - pad_w_1
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        inp_dec_level1 = self.reduce_chan_level1_norm(inp_dec_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = inp_dec_level1

        if self.training and self.gradient_checkpointing:

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            
            for block in self.decoder_level_1:
                out_dec_level1 = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    out_dec_level1,
                    attention_bias,
                    encoder_hidden_states_1,
                    encoder_attention_mask,
                    timestep_1,
                    cross_attention_kwargs,
                    class_labels,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
        else:
            for block in self.decoder_level_1:
                out_dec_level1 = block(
                    out_dec_level1,
                    attention_mask=attention_bias,
                    encoder_hidden_states=encoder_hidden_states_1,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep_1,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )

        assert not torch.any(torch.isnan(out_dec_level1)), 'after out_dec_level1'
        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=out_dec_level1,
            timestep=timestep_1,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep_1,
            num_frames=frame, 
            height=height,
            width=width,
        )  # b c t h w
        
        assert not torch.any(torch.isnan(output)), 'after output'
        frame, height, width = frame * self.config.patch_size_t, height * self.config.patch_size - pad_h_0, width * self.config.patch_size - pad_w_0
        output = output[:, :, :frame, :height, :width]
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num):
        # batch_size = hidden_states.shape[0]
        hidden_states = self.pos_embed(hidden_states.to(self.dtype), frame)

        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )
        timestep_1, embedded_timestep_1 = self.adaln_single_1(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d
        timestep_2, embedded_timestep_2 = self.adaln_single_2(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d
        timestep_3, embedded_timestep_3 = self.adaln_single_3(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states_1 = self.caption_projection_1(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d
        encoder_hidden_states_1 = rearrange(encoder_hidden_states_1[:, :1], 'b 1 l d -> (b 1) l d')
        encoder_hidden_states_2 = self.caption_projection_2(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d
        encoder_hidden_states_2 = rearrange(encoder_hidden_states_2[:, :1], 'b 1 l d -> (b 1) l d')
        encoder_hidden_states_3 = self.caption_projection_3(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d
        encoder_hidden_states_3 = rearrange(encoder_hidden_states_3[:, :1], 'b 1 l d -> (b 1) l d')


        return hidden_states, encoder_hidden_states_1, encoder_hidden_states_2, encoder_hidden_states_3, \
            timestep_1, timestep_2, timestep_3, embedded_timestep_1, embedded_timestep_2, embedded_timestep_3

    
    
    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, class_labels, embedded_timestep, num_frames, height=None, width=None
    ):  
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

        # # unpatchify
        # hidden_states = hidden_states.reshape(
        #     shape=(-1, num_frames, height, width, self.out_channels)
        # )
        # output = torch.einsum("nthwc->ncthw", hidden_states)
        # return output
            # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, num_frames, height, width, self.config.patch_size_t, self.config.patch_size, self.config.patch_size, self.config.out_channels)
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.config.out_channels, num_frames * self.config.patch_size_t, height * self.config.patch_size, width * self.config.patch_size)
        )
        # import ipdb;ipdb.set_trace()
        # if output.shape[2] % 2 == 0:
        #     output = output[:, :, 1:]
        return output


def UDiTT2V_S_111(**kwargs):
    return UDiTT2V(depth=[2, 5, 8, 5, 2], attention_head_dim=24, num_attention_heads=16, patch_size_t=1, patch_size=1, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_S_122(**kwargs):
    return UDiTT2V(depth=[2, 5, 8, 5, 2], attention_head_dim=24, num_attention_heads=16, patch_size_t=1, patch_size=2, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_B_111(**kwargs):
    return UDiTT2V(depth=[2, 5, 10, 5, 2], attention_head_dim=24, num_attention_heads=24, patch_size_t=1, patch_size=1, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_B_122(**kwargs):
    return UDiTT2V(depth=[2, 5, 10, 5, 2], attention_head_dim=24, num_attention_heads=24, patch_size_t=1, patch_size=2, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_L_111(**kwargs):
    return UDiTT2V(depth=[4, 8, 12, 8, 4], attention_head_dim=24, num_attention_heads=32, patch_size_t=1, patch_size=1, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_L_211(**kwargs):
    return UDiTT2V(depth=[4, 8, 12, 8, 4], attention_head_dim=24, num_attention_heads=32, patch_size_t=2, patch_size=1, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_L_122(**kwargs):
    return UDiTT2V(depth=[4, 8, 12, 8, 4], attention_head_dim=24, num_attention_heads=32, patch_size_t=1, patch_size=2, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_L_222(**kwargs):
    return UDiTT2V(depth=[4, 8, 12, 8, 4], attention_head_dim=24, num_attention_heads=32, patch_size_t=2, patch_size=2, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)

def UDiTT2V_XL_111(**kwargs):
    return UDiTT2V(depth=[4, 8, 12, 8, 4], attention_head_dim=24, num_attention_heads=48, patch_size_t=1, patch_size=1, 
                   mlp_ratio=2, norm_type="ada_norm_single", caption_channels=4096, **kwargs)


UDiT_models = {
    "UDiTT2V-S/111": UDiTT2V_S_111,  # 0.4B    0.7B if video
    "UDiTT2V-S/122": UDiTT2V_S_122,  # 0.4B    0.7B if video
    "UDiTT2V-B/111": UDiTT2V_B_111,  # 1.0B    1.6B if video
    "UDiTT2V-B/122": UDiTT2V_B_122,  # 1.0B    1.6B if video
    "UDiTT2V-L/111": UDiTT2V_L_111,  # 2.2B    3.3B if video
    "UDiTT2V-L/211": UDiTT2V_L_211,  # 2.2B    3.3B if video
    "UDiTT2V-L/122": UDiTT2V_L_122,  # 2.2B    3.3B if video
    "UDiTT2V-L/222": UDiTT2V_L_222,  # 2.2B    3.3B if video
    "UDiTT2V-XL/111": UDiTT2V_XL_111,  # 5.0B    7.4B if video
}

UDiT_models_class = {
    "UDiTT2V-S/111": UDiTT2V,
    "UDiTT2V-S/122": UDiTT2V,
    "UDiTT2V-B/111": UDiTT2V,
    "UDiTT2V-B/122": UDiTT2V,
    "UDiTT2V-L/111": UDiTT2V,
    "UDiTT2V-L/211": UDiTT2V,
    "UDiTT2V-L/122": UDiTT2V,
    "UDiTT2V-L/222": UDiTT2V,
    "UDiTT2V-XL/111": UDiTT2V,
}


if __name__ == '__main__':
    import sys
    from copy import deepcopy
    from opensora.models.causalvideovae import ae_channel_config, ae_stride_config
    from opensora.models.causalvideovae import getae, getae_wrapper
    from opensora.models.causalvideovae import CausalVAEModelWrapper

    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_4x8x8', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 240, 
        'max_width': 320, 
        'num_frames': 1, 
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
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


    
    model = UDiTT2V_L_122(in_channels=c, 
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
                            use_rope=args.use_rope).to(device)

    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9} B')


    model_state_dict = model.state_dict()
    pretrained = "/storage/ongoing/new/Open-Sora-Plan/bs2_20node_73000k_480p_61x480p_lr5e-5_snr5_noioff0.02_ema_rope_uditultra122_qknorm_ds222_mt5xxl_sucai288w/checkpoint-11500/model_ema/diffusion_pytorch_model.safetensors"
    try:
        if 'safetensors' in pretrained:  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(pretrained, device="cpu")
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
            # if checkpoint['pos_embed.proj.weight'].shape != model.pos_embed.proj.weight.shape and checkpoint['pos_embed.proj.weight'].ndim == 4:
            #     logger.info(f"Resize pos_embed, {checkpoint['pos_embed.proj.weight'].shape} -> {model.pos_embed.proj.weight.shape}")
            #     repeat = model.pos_embed.proj.weight.shape[2]
            #     checkpoint['pos_embed.proj.weight'] = checkpoint['pos_embed.proj.weight'].unsqueeze(2).repeat(1, 1, repeat, 1, 1) / float(repeat)
                # del checkpoint['proj_out.weight'], checkpoint['proj_out.bias']
        else:  # latest stage training weight
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
        # import ipdb;ipdb.set_trace()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')
    except Exception as e:
        print(e)

    # import sys;sys.exit()
    # try:
    #     path = "bs32_1node_480p_lr1e-4_snr5_noioff0.02_ema_uditultra22_ds22_mt5xxl/checkpoint-500/model/diffusion_pytorch_model.safetensors"
    #     from safetensors.torch import load_file as safe_load
    #     ckpt = safe_load(path, device="cpu")
    #     new_ckpt = {}
    #     k_size = 3
    #     t_stride = 1
    #     for k, v in ckpt.items():
    #         if 'pos_embed.proj.weight' in k:
    #             new_v = v.unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
    #         elif 'attn1.downsampler.layer.weight' in k:
    #             new_v = v.unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 768, 4, 3, 3 -> 768, 4, 3, 3, 3
    #         elif 'body.0.weight' in k and 'down' in k:
    #             in_c = v.shape[0]
    #             new_v = v[:in_c//2].unsqueeze(-3).repeat(1, 1, k_size, 1, 1)  # 384, 768, 3, 3 -> 192, 768, 3, 3, 3
    #         elif 'body.0.weight' in k and 'up' in k:
    #             new_v = v.unsqueeze(-3).repeat(2, 1, k_size, 1, 1)  # 6144, 3072, 3, 3 -> 12288, 3072, 3, 3, 3
    #         elif 'proj_out' in k:
    #             if 'weight' in k:
    #                 new_v = v.repeat(t_stride, 1)  # 16, 768 -> 32, 768
    #             elif 'bias' in k:
    #                 new_v = v.repeat(t_stride)  # 16 -> 32
    #         else:
    #             new_v = v
    #         new_ckpt[k] = new_v
    #     msg = model.load_state_dict(new_ckpt, strict=False)
    #     # print(msg)
    # except Exception as e:
    #     print(e)
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1+args.use_image_num, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1+args.use_image_num, args.model_max_length)).to(device)  # B L or B 1+num_images L
    timestep = torch.randint(0, 1000, (b,), device=device)
    model_kwargs = dict(hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask, 
                        encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, timestep=timestep)
    with torch.no_grad():
        output = model(**model_kwargs)[0]
    import ipdb;ipdb.set_trace()
    print(output.shape)
    



    # from peft import LoraConfig, PeftModel, get_peft_model
    # from opensora.utils.lora_utils import EMAModel_LoRA, maybe_zero_3, get_peft_state_maybe_zero_3
    # lora_save_path = '/storage/ongoing/new/Open-Sora-Plan/debug_lora/model_lora'
    # ema_lora_save_path = '/storage/ongoing/new/Open-Sora-Plan/debug_lora/ema_model_lora'
    # origin_model_path = '/storage/ongoing/new/Open-Sora-Plan/bs16_4node_240p_lr1e-4_snr5_noioff0.02_ema_rope_uditultra22_ds22_mt5xxl/checkpoint-500/model_ema'
    # model = UDiTUltraT2V.from_pretrained(origin_model_path)
    # lora_config = LoraConfig(
    #     r=64,
    #     lora_alpha=64,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # model_lora = get_peft_model(model, lora_config)
    # # --------------------ema lora_model----------------------------------
    # # create ema lora_model
    # ema_model = deepcopy(model_lora)
    # ema_model_lora = EMAModel_LoRA(lora_config, parameters=ema_model.parameters(), update_after_step=0, 
    #                                model_cls=UDiTUltraT2V, model_config=ema_model.config)
    # ema_model_lora.save_pretrained(ema_lora_save_path)
    # ema_model_load_lora = EMAModel_LoRA.from_pretrained(ema_lora_save_path, UDiTUltraT2V, lora_config, origin_model_path)
    # ema_model_lora.load_state_dict(ema_model_load_lora.state_dict())
    # ema_model_lora.to(device)

    # # -----------------lora model---------------------------------
    # # get lora weight
    # model_lora.save_pretrained(lora_save_path)
    # # ----------------load lora model------------------------------
    # # load lora weight
    # model = UDiTUltraT2V.from_pretrained(origin_model_path)
    # import ipdb;ipdb.set_trace()
    # model_load_lora = PeftModel.from_pretrained(model, lora_save_path)
    # for k, v in model_load_lora.state_dict().items():
    #     assert torch.allclose(v, model_lora.state_dict()[k])
    # # for k, v in zip(ema_model_lora.shadow_params, model_lora.parameters()):
    # #     assert torch.allclose(v, k)
    # print('Merging LoRA weights...')
    # import ipdb;ipdb.set_trace()
    # model_load_lora_merge = model_load_lora.merge_and_unload()
    # with torch.no_grad():
    #     output = model_load_lora_merge(**model_kwargs)
    # print(output[0].shape)




