from typing import Optional, Tuple
from einops import rearrange, repeat
from torch import nn
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, PixArtAlphaTextProjection
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormSingle
from diffusers.models.attention import FeedForward
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from mindspeed_mm.models.common.attention import MultiHeadSparseAttentionSBH
from mindspeed_mm.models.common.motion import MotionAdaLayerNormSingle


class VideoDiTSparse(ModelMixin, ConfigMixin):
    """
    A video dit model for video generation. can process both standard continuous images of shape
    (batch_size, num_channels, width, height) as well as quantized image embeddings of shape
    (batch_size, num_image_vectors). Define whether input is continuous or discrete depending on config.

    Args:
        num_layers: The number of layers for VideoDiTBlock.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in· the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        input_size: The shape of the latents (specify if the input is discrete).
        patch_size: The shape of the patchs.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_type: can be 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'.
        num_embeds_ada_norm: The number of diffusion steps used during training. Pass if at least one of the norm_layers is
                             `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings
                             that are added to the hidden states.
        norm_elementswise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of he normalization.
        use_rope: Whether to use rope in attention block.
        interpolation_scale: The scale for interpolation.
    """

    @register_to_config
    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        interpolation_scale: Tuple[float] = None,
        sparse1d: bool = False,
        sparse_n: int = 2,
        **kwargs
    ):
        super().__init__()
        args = get_args()
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        self.out_channels = in_channels if out_channels is None else out_channels
        self.config.hidden_size = self.config.num_heads * self.config.head_dim

        self._init_patched_inputs()

    def _init_patched_inputs(self):

        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels, hidden_size=self.config.hidden_size
        )
        self.motion_projection = MotionAdaLayerNormSingle(self.config.hidden_size)

        self.pos_embed = PatchEmbed2D(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )

        self.videodit_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.config.hidden_size,
                    self.config.num_heads,
                    self.config.head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    interpolation_scale=self.config.interpolation_scale,
                    sparse1d=self.config.sparse1d if _ > 1 and _ < 30 else False,
                    sparse_n=self.config.sparse_n,
                    sparse_group=_ % 2 == 1,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.config.hidden_size) / self.config.hidden_size ** 0.5)
        self.proj_out = nn.Linear(
            self.config.hidden_size,
            self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.config.out_channels
        )
        self.adaln_single = AdaLayerNormSingle(self.config.hidden_size)

    def prepare_sparse_mask(self, attention_mask, encoder_attention_mask, sparse_n):
        attention_mask = attention_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        l = attention_mask.shape[-1]
        if l % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - l % (sparse_n * sparse_n)

        attention_mask_sparse = F.pad(attention_mask, (0, pad_len, 0, 0), value=-9980.0)
        attention_mask_sparse_1d = rearrange(
            attention_mask_sparse,
            'b 1 1 (g k) -> (k b) 1 1 g',
            k=sparse_n
        )
        attention_mask_sparse_1d_group = rearrange(
            attention_mask_sparse,
            'b 1 1 (n m k) -> (m b) 1 1 (n k)',
            m=sparse_n,
            k=sparse_n
        )
        encoder_attention_mask_sparse = encoder_attention_mask.repeat(sparse_n, 1, 1, 1)

        def get_attention_mask(mask, repeat_num):
            mask = mask.to(torch.bool)
            mask = mask.repeat(1, 1, repeat_num, 1)
            return mask

        attention_mask_sparse_1d = get_attention_mask(
            attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1]
        )

        attention_mask_sparse_1d_group = get_attention_mask(
            attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1]
        )

        encoder_attention_mask_sparse_1d = get_attention_mask(
            encoder_attention_mask_sparse, attention_mask_sparse_1d.shape[-1]
        )

        encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d
        return {
            False: (attention_mask_sparse_1d, encoder_attention_mask_sparse_1d),
            True: (attention_mask_sparse_1d_group, encoder_attention_mask_sparse_1d_group)
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        motion_score: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, c, frames, h, w = hidden_states.shape
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
            # b, frames, h, w -> a video
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
        frames = ((frames - 1) // self.config.patch_size_t + 1) if frames % 2 == 1 else frames // self.config.patch_size_t  # patchfy
        height, width = hidden_states.shape[-2] // self.config.patch_size, hidden_states.shape[
            -1] // self.config.patch_size

        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, motion_score, batch_size, frames
        )

        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

        sparse_mask = {}
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = self.prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n)

        # 2. Blocks
        for i, block in enumerate(self.videodit_blocks):
            if i > 1 and i < 30:
                attention_mask, encoder_attention_mask = sparse_mask[block.self_atten.sparse_n][block.self_atten.sparse_group]
            else:
                attention_mask, encoder_attention_mask = sparse_mask[1][block.self_atten.sparse_group]
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                frames=frames,
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
            num_frames=frames,
            height=height,
            width=width,
        )  # b c t h w

        return output

    def _get_block(self, layer_number):
        return self.videodit_blocks[layer_number]

    def _checkpointed_forward(
        self,
        latents,
        video_mask,
        prompt, 
        prompt_mask,
        timestep,
        class_labels,
        frames,
        height,
        width):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_block(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_
            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_num = 0
            while layer_num < self.num_layers:
                latents = tensor_parallel.checkpoint(
                    custom(layer_num, layer_num + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    latents,
                    prompt,
                    video_mask,
                    prompt_mask,
                    timestep,
                    class_labels,
                    frames,
                    height,
                    width
                )
                layer_num += self.recompute_num_layers
        elif self.recompute_method == "block":
            for layer_num in range(self.num_layers):
                if layer_num < self.recompute_num_layers:
                    latents = tensor_parallel.checkpoint(
                        custom(layer_num, layer_num + 1),
                        self.distribute_saved_activations,
                        latents,
                        prompt,
                        video_mask,
                        prompt_mask,
                        timestep,
                        class_labels,
                        frames,
                        height,
                        width
                    )
                else:
                    block = self._get_block(layer_num)
                    latents = block(
                        latents,
                        video_mask=video_mask,
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        timestep=timestep,
                        class_labels=class_labels,
                        frames=frames,
                        height=height,
                        width=width
                    )
        else:
            raise ValueError("Invalid activation recompute method.")
        return latents

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, motion_score, batch_size,
                                   frames):

        hidden_states = self.pos_embed(hidden_states.to(self.dtype))

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        motion_embed = self.motion_projection(motion_score, batch_size=batch_size, hidden_dtype=self.dtype)  # b 6d
        timestep = timestep + motion_embed

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
            shape=(
            -1, num_frames, height, width, self.config.patch_size_t, self.config.patch_size, self.config.patch_size,
            self.out_channels)
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   num_frames * self.config.patch_size_t, height * self.config.patch_size,
                   width * self.config.patch_size)
        )
        return output

class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True,
            interpolation_scale: Tuple[int] = (1, 1, 1),
            sparse1d: bool = False,
            sparse_n: int = 2,
            sparse_group: bool = False,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.self_atten = MultiHeadSparseAttentionSBH(
            query_dim=dim,
            key_dim=cross_attention_dim if only_cross_attention else None,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            interpolation_scale=interpolation_scale,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=False,
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.cross_atten = MultiHeadSparseAttentionSBH(
            query_dim=dim,
            key_dim=cross_attention_dim if not double_self_attention else None,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            interpolation_scale=interpolation_scale,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=True,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Scale-shift.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            frames: int = None,
            height: int = None,
            width: int = None,
    ) -> torch.FloatTensor:
        # 0. Self-Attention
        batch_size = hidden_states.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
        ).chunk(6, dim=0)

        norm_hidden_states = self.norm1(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_output = self.self_atten(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask, frames=frames, height=height, width=width,
        )

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        norm_hidden_states = hidden_states

        attn_output = self.cross_atten(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask, frames=frames, height=height, width=width,
        )
        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding but with video"""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)
        latent = rearrange(latent, '(b t) c h w -> b (t h w) c', b=b)
        return latent