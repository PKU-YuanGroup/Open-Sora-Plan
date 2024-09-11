from typing import Optional, Tuple, Dict

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, PixArtAlphaTextProjection
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormSingle
from diffusers.models.attention import FeedForward
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args

from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.embeddings.patch_embeddings import VideoPatchEmbed2D
from mindspeed_mm.models.common.attention import MultiHeadAttentionBSH, ParallelMultiHeadAttentionSBH


class VideoDiT(MultiModalModule):
    """
    A video dit model for video generation. can process both standard continuous images of shape
    (batch_size, num_channels, width, height) as well as quantized image embeddings of shape
    (batch_size, num_image_vectors). Define whether input is continuous or discrete depending on config.

    Args:
        num_layers: The number of layers for VideoDiTBlock.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in the input (specify if the input is continuous).
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

    def __init__(
        self,
        num_layers: int = 1,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        input_size: Tuple[int] = None,
        patch_size: Tuple[int] = None,
        activation_fn: str = "geglu",
        norm_type: str = "layer_norm",
        num_embeds_ada_norm: Optional[int] = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        use_rope: bool = False,
        interpolation_scale: Tuple[float] = None,
        **kwargs
    ):
        super().__init__(config=None)
        # Validate inputs and init args.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )
        self.patch_size_t, self.patch_size_h, self.patch_size_w = patch_size
        self.norm_type = norm_type
        self.out_channels = out_channels
        self.num_layers = num_layers
        inner_dim = num_heads * head_dim

        args = get_args()
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        if mpu.get_context_parallel_world_size() > 1:
            self.enable_sequence_parallelism = True
        else:
            self.enable_sequence_parallelism = False

        # Initialize blocks
        # Init PatchEmbed
        self.pos_embed = VideoPatchEmbed2D(
            num_frames=input_size[0],
            height=input_size[1],
            width=input_size[2],
            patch_size_t=self.patch_size_t,
            patch_size=self.patch_size_h,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=(interpolation_scale[1], interpolation_scale[2]),
            interpolation_scale_t=interpolation_scale[0],
            use_abs_pos=not use_rope,
        )
        # Init VideoDiTBlock
        self.videodit_blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    dim=inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    use_rope=use_rope,
                    interpolation_scale=interpolation_scale,
                    enable_sequence_parallelism=self.enable_sequence_parallelism,
                )
                for _ in range(num_layers)
            ]
        )
        # Init Norm
        if norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(inner_dim, self.patch_size_t * self.patch_size_h * self.patch_size_w * self.out_channels)
        elif norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
            self.proj_out = nn.Linear(inner_dim, self.patch_size_t * self.patch_size_h * self.patch_size_w * self.out_channels)
        self.adaln_single = None
        if norm_type == "ada_norm_single":
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)
        # Init Projection
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            latents: Shape (batch size, num latent pixels) if discrete, shape (batch size, channel, height, width) if continuous.
            timestep: Used to indicate denoising step. Optional timestep to be applied as an embedding in AdaLayerNorm.
            prompt: Conditional embeddings for cross attention layer.
            video_mask: An attention mask of shape (batch, key_tokens) is applied to latents.
            prompt_mask: Cross-attention mask applied to prompt.
            added_cond_kwargs: resolution or aspect_ratio.
            class_labels: Used to indicate class labels conditioning.
            use_image_num: The number of images use for trainning.
        """
        b, _, t, _, _ = latents.shape
        frames = t - use_image_num
        vid_mask, img_mask = None, None
        prompt_mask = prompt_mask.view(b, -1, prompt.shape[-1])
        if video_mask is not None and video_mask.ndim == 4:
            video_mask = video_mask.to(self.dtype)
            vid_mask = video_mask[:, :frames]  # [b, frames, h, w]
            img_mask = video_mask[:, frames:]  # [b, use_image_num, h, w]

            if vid_mask.numel() > 0:
                vid_mask_first_frame = vid_mask[:, :1].repeat(1, self.patch_size_t - 1, 1, 1)
                vid_mask = torch.cat([vid_mask_first_frame, vid_mask], dim=1)
                vid_mask = vid_mask.unsqueeze(1)  # [b, 1, t, h, w]
                vid_mask = F.max_pool3d(vid_mask, kernel_size=(self.patch_size_t, self.patch_size_h, self.patch_size_w),
                                        stride=(self.patch_size_t, self.patch_size_h, self.patch_size_w))
                vid_mask = rearrange(vid_mask, 'b 1 t h w -> (b 1) 1 (t h w)')
            if img_mask.numel() > 0:
                img_mask = F.max_pool2d(img_mask, kernel_size=(self.patch_size_h, self.patch_size_w),
                                        stride=(self.patch_size_h, self.patch_size_w))
                img_mask = rearrange(img_mask, 'b i h w -> (b i) 1 (h w)')

            vid_mask = (1 - vid_mask.bool().to(self.dtype)) * -10000.0 if vid_mask.numel() > 0 else None
            img_mask = (1 - img_mask.bool().to(self.dtype)) * -10000.0 if img_mask.numel() > 0 else None

            if frames == 1 and use_image_num == 0 and not self.enable_sequence_parallelism:
                img_mask = vid_mask
                vid_mask = None
        # convert prompt_mask to a bias the same way we do for video_mask
        if prompt_mask is not None and prompt_mask.ndim == 3:
            prompt_mask = (1 - prompt_mask.to(self.dtype)) * -10000.0
            in_t = prompt_mask.shape[1]
            prompt_vid_mask = prompt_mask[:, :in_t - use_image_num]
            prompt_vid_mask = rearrange(prompt_vid_mask, 'b 1 l -> (b 1) 1 l') if prompt_vid_mask.numel() > 0 else None

            prompt_img_mask = prompt_mask[:, in_t - use_image_num:]
            prompt_img_mask = rearrange(prompt_img_mask, 'b i l -> (b i) 1 l') if prompt_img_mask.numel() > 0 else None

            if frames == 1 and use_image_num == 0 and not self.enable_sequence_parallelism:
                prompt_img_mask = prompt_vid_mask
                prompt_vid_mask = None

        if vid_mask is not None:
            vid_mask = vid_mask.bool().repeat(1, vid_mask.shape[-1], 1)
            prompt_vid_mask = prompt_vid_mask.bool().repeat(1, vid_mask.shape[-2], 1)
        if img_mask is not None:
            img_mask = img_mask.bool().repeat(1, img_mask.shape[-1], 1)
            prompt_img_mask = prompt_img_mask.bool().repeat(1, img_mask.shape[-2], 1)

        # 1. Input
        frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
        height, width = latents.shape[-2] // self.patch_size_h, latents.shape[-1] // self.patch_size_w

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, \
            embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
            latents, prompt, timestep, added_cond_kwargs, b, frames, use_image_num
        )

        if self.enable_sequence_parallelism and latents_vid is not None and prompt_vid is not None:
            latents_vid = rearrange(latents_vid, 'b s h -> s b h', b=b).contiguous()
            prompt_vid = rearrange(prompt_vid, 'b s h -> s b h', b=b).contiguous()
            timestep_vid = timestep_vid.view(latents_vid.shape[1], 6, -1).transpose(0, 1).contiguous()

            latents_vid = split_forward_gather_backward(latents_vid, mpu.get_context_parallel_group(), dim=0,
                                                        grad_scale='down')

        frames = torch.tensor(frames)
        height = torch.tensor(height)
        width = torch.tensor(width)
        if self.recompute_granularity == "full":
            if latents_vid is not None:
                latents_vid = self._checkpointed_forward(
                    latents_vid,
                    video_mask=vid_mask,
                    prompt=prompt_vid,
                    prompt_mask=prompt_vid_mask,
                    timestep=timestep_vid,
                    class_labels=class_labels,
                    frames=frames,
                    height=height,
                    width=width
                )
            if latents_img is not None:
                latents_img = self._checkpointed_forward(
                    latents_img,
                    video_mask=img_mask,
                    prompt=prompt_img,
                    prompt_mask=prompt_img_mask,
                    timestep=timestep_img,
                    class_labels=class_labels,
                    frames=torch.tensor(1),
                    height=height,
                    width=width
                )
        else:
            for block in self.videodit_blocks:
                if latents_vid is not None:
                    latents_vid = block(
                        latents_vid,
                        video_mask=vid_mask,
                        prompt=prompt_vid,
                        prompt_mask=prompt_vid_mask,
                        timestep=timestep_vid,
                        class_labels=class_labels,
                        frames=frames,
                        height=height,
                        width=width
                    )
                if latents_img is not None:
                    latents_img = block(
                        latents_img,
                        video_mask=img_mask,
                        prompt=prompt_img,
                        prompt_mask=prompt_img_mask,
                        timestep=timestep_img,
                        class_labels=class_labels,
                        frames=torch.tensor(1),
                        height=height,
                        width=width
                    )

        if self.enable_sequence_parallelism and latents_vid is not None:
            latents_vid = rearrange(latents_vid, 's b h -> b s h', b=b).contiguous()
            latents_vid = gather_forward_split_backward(latents_vid, mpu.get_context_parallel_group(), dim=1,
                                                        grad_scale='up')

        # 3. Output
        output_vid, output_img = None, None
        if latents_vid is not None:
            output_vid = self._get_output_for_patched_inputs(
                latents=latents_vid,
                timestep=timestep_vid,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_vid,
                num_frames=frames,
                height=height,
                width=width,
            )  # [b, c, t, h, w]
        if latents_img is not None:
            output_img = self._get_output_for_patched_inputs(
                latents=latents_img,
                timestep=timestep_img,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_img,
                num_frames=1,
                height=height,
                width=width,
            )  # [b, c, 1, h, w]
            if use_image_num != 0:
                output_img = rearrange(output_img, '(b i) c 1 h w -> b c i h w', i=use_image_num)

        if output_vid is not None and output_img is not None:
            output = torch.cat([output_vid, output_img], dim=2)
        elif output_vid is not None:
            output = output_vid
        elif output_img is not None:
            output = output_img
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

    def _operate_on_patched_inputs(self, latents, prompt, timestep, added_cond_kwargs, batch_size, frames,
                                   use_image_num):
        latents_vid, latents_img = self.pos_embed(latents.to(self.dtype), frames)
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        prompt_vid, prompt_img = None, None

        if self.adaln_single is not None:
            timestep, embedded_timestep = self.adaln_single(timestep, added_cond_kwargs, batch_size=batch_size,
                                                            hidden_dtype=self.dtype)
            if latents_vid is None:
                timestep_img = timestep
                embedded_timestep_img = embedded_timestep
            else:
                timestep_vid = timestep
                embedded_timestep_vid = embedded_timestep
                if latents_img is not None:
                    timestep_img = repeat(timestep, 'b d -> (b i) d', i=use_image_num).contiguous()
                    embedded_timestep_img = repeat(embedded_timestep, 'b d -> (b i) d', i=use_image_num).contiguous()

        if self.caption_projection is not None:
            prompt = self.caption_projection(prompt)
            if latents_vid is None:
                prompt_img = rearrange(prompt, 'b 1 l d -> (b 1) l d')
            else:
                prompt_vid = rearrange(prompt[:, :1], 'b 1 l d -> (b 1) l d')
                if latents_img is not None:
                    prompt_img = rearrange(prompt[:, 1:], 'b i l d -> (b i) l d')

        return latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img

    def _get_output_for_patched_inputs(self, latents, timestep, class_labels, embedded_timestep, num_frames,
                                       height=None, width=None):
        if self.norm_type != "ada_norm_single":
            conditioning = self.videodit_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=self.dtype)
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            latents = self.norm_out(latents) * (1 + scale[:, None]) + shift[:, None]
            latents = self.proj_out_2(latents)
        elif self.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            latents = self.norm_out(latents)
            # Modulation
            latents = latents * (1 + scale) + shift
            latents = self.proj_out(latents)
            latents = latents.squeeze(1)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(latents.shape[1] ** 0.5)
        latents = latents.reshape(shape=(-1, num_frames, height, width, self.patch_size_t,
                                         self.patch_size_h, self.patch_size_w, self.out_channels))
        latents = torch.einsum("nthwopqc->nctohpwq", latents)
        output = latents.reshape(shape=(-1, self.out_channels, num_frames * self.patch_size_t,
                                        height * self.patch_size_h, width * self.patch_size_w))
        return output


class VideoDiTBlock(nn.Module):
    """
    A basic dit block for video generation.

    Args:
        dim: The number out channels in the input and output.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_type: can be 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'.
        num_embeds_ada_norm: The number of diffusion steps used during training. Pass if at least one of the norm_layers is
                             `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings
                             that are added to the hidden states.
        norm_elementswise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of he normalization.
        interpolation_scale: The scale for interpolation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        use_rope: bool = False,
        interpolation_scale: Tuple[float] = None,
        enable_sequence_parallelism: bool = False,
    ):
        super().__init__()
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError("If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined.")
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define three blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.enable_sequence_parallelism = enable_sequence_parallelism
        if self.enable_sequence_parallelism:
            attention = ParallelMultiHeadAttentionSBH
        else:
            attention = MultiHeadAttentionBSH

        self.self_atten = attention(
            query_dim=dim,
            key_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            use_rope=use_rope,
            interpolation_scale=interpolation_scale
        )

        # 2. Cross-Attn
        if norm_type == "ada_norm":
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm2 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.cross_atten = attention(
            query_dim=dim,
            key_dim=cross_attention_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias
        )

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Scale-shift.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        latents: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        timestep: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        frames: torch.int64 = None, 
        height: torch.int64 = None, 
        width: torch.int64 = None, 
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # 1. Self-Attention
        frames = frames.item()
        height = height.item()
        width = width.item()
        batch_size = latents.shape[0]
        if self.norm_type == "ada_norm":
            norm_latents = self.norm1(latents, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_latents, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                latents, timestep, class_labels, hidden_dtype=latents.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_latents = self.norm1(latents)
        elif self.norm_type == "ada_norm_continuous":
            norm_latents = self.norm1(latents, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            if self.enable_sequence_parallelism:
                batch_size = latents.shape[1]
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
                ).chunk(6, dim=0)
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
            norm_latents = self.norm1(latents)
            norm_latents = norm_latents * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_latents = self.pos_embed(norm_latents)

        attn_output = self.self_atten(
            query=norm_latents,
            key=None,
            mask=video_mask,
            frames=frames,
            height=height,
            width=width
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        latents = attn_output + latents
        if latents.ndim == 4:
            latents = latents.squeeze(1)

        # 2. Cross-Attention
        if self.norm_type == "ada_norm":
            norm_latents = self.norm2(latents, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_latents = self.norm2(latents)
        elif self.norm_type == "ada_norm_single":
            norm_latents = latents
        elif self.norm_type == "ada_norm_continuous":
            norm_latents = self.norm2(latents, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_latents = self.pos_embed(norm_latents)

        attn_output = self.cross_atten(
            query=norm_latents,
            key=prompt,
            mask=prompt_mask
        )
        latents = attn_output + latents

        # 3. Feed-forward
        if self.norm_type == "ada_norm_continuous":
            norm_latents = self.norm3(latents, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_latents = self.norm3(latents)
        if self.norm_type == "ada_norm_zero":
            norm_latents = norm_latents * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self.norm_type == "ada_norm_single":
            norm_latents = self.norm2(latents)
            norm_latents = norm_latents * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_latents)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        latents = ff_output + latents
        if latents.ndim == 4:
            latents = latents.squeeze(1)
        return latents

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim
