from typing import Optional, Tuple

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.normalization import AdaLayerNormSingle
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.embeddings import PatchEmbed2D
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.attention import MultiHeadSparseAttentionSBH
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward


class VideoDitSparse(MultiModalModule):
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
        patch_size: The shape of the patchs.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_elementwise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of the normalization.
        interpolation_scale: The scale for interpolation.
    """

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
        super().__init__(config=None)
        args = get_args()
        self.gradient_checkpointing = True
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        inner_dim = num_heads * head_dim
        self.num_layers = num_layers
        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.pos_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
        )

        self.videodit_sparse_blocks = nn.ModuleList(
            [
                VideoDiTSparseBlock(
                    inner_dim,
                    num_heads,
                    head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    interpolation_scale=interpolation_scale,
                    sparse1d=sparse1d if 1 < _ < 30 else False,
                    sparse_n=sparse_n,
                    sparse_group=_ % 2 == 1,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)
        self.adaln_single = AdaLayerNormSingle(inner_dim)

    def prepare_sparse_mask(self, video_mask, prompt_mask, sparse_n):
        video_mask = video_mask.unsqueeze(1)
        prompt_mask = prompt_mask.unsqueeze(1)
        _len = video_mask.shape[-1]
        if _len % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - _len % (sparse_n * sparse_n)

        video_mask_sparse = F.pad(video_mask, (0, pad_len, 0, 0), value=-9980.0)
        video_mask_sparse_1d = rearrange(
            video_mask_sparse,
            'b 1 1 (g k) -> (k b) 1 1 g',
            k=sparse_n
        )
        video_mask_sparse_1d_group = rearrange(
            video_mask_sparse,
            'b 1 1 (n m k) -> (m b) 1 1 (n k)',
            m=sparse_n,
            k=sparse_n
        )
        prompt_mask_sparse = prompt_mask.repeat(sparse_n, 1, 1, 1)

        def get_attention_mask(mask, repeat_num):
            mask = mask.to(torch.bool)
            mask = mask.repeat(1, 1, repeat_num, 1)
            return mask

        video_mask_sparse_1d = get_attention_mask(video_mask_sparse_1d, video_mask_sparse_1d.shape[-1])
        video_mask_sparse_1d_group = get_attention_mask(
            video_mask_sparse_1d_group, video_mask_sparse_1d_group.shape[-1]
        )
        prompt_mask_sparse_1d = get_attention_mask(
            prompt_mask_sparse, video_mask_sparse_1d.shape[-1]
        )
        prompt_mask_sparse_1d_group = prompt_mask_sparse_1d

        return {
            False: (video_mask_sparse_1d, prompt_mask_sparse_1d),
            True: (video_mask_sparse_1d_group, prompt_mask_sparse_1d_group)
        }

    def forward(
        self,
        latents: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, c, frames, h, w = latents.shape
        prompt_mask = prompt_mask.view(batch_size, -1, prompt_mask.shape[-1])
        if self.training and mpu.get_context_parallel_world_size() > 1:
            frames //= mpu.get_context_parallel_world_size()
            latents = split_forward_gather_backward(latents, mpu.get_context_parallel_group(), dim=2,
                                                    grad_scale='down')
            prompt = split_forward_gather_backward(prompt, mpu.get_context_parallel_group(),
                                                   dim=2, grad_scale='down')

        if video_mask is not None and video_mask.ndim == 4:
            video_mask = video_mask.to(self.dtype)

            video_mask = video_mask.unsqueeze(1)  # b 1 t h w
            video_mask = F.max_pool3d(
                video_mask,
                kernel_size=(self.patch_size_t, self.patch_size, self.patch_size),
                stride=(self.patch_size_t, self.patch_size, self.patch_size)
            )
            video_mask = rearrange(video_mask, 'b 1 t h w -> (b 1) 1 (t h w)')
            video_mask = (1 - video_mask.bool().to(self.dtype)) * -10000.0

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if prompt_mask is not None and prompt_mask.ndim == 3:
            # b, 1, l
            prompt_mask = (1 - prompt_mask.to(self.dtype)) * -10000.0

        # 1. Input
        frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
        height, width = latents.shape[-2] // self.patch_size, latents.shape[-1] // self.patch_size

        latents, prompt, timestep, embedded_timestep = self._operate_on_patched_inputs(
            latents, prompt, timestep, batch_size
        )

        latents = rearrange(latents, 'b s h -> s b h', b=batch_size).contiguous()
        prompt = rearrange(prompt, 'b s h -> s b h', b=batch_size).contiguous()
        timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

        sparse_mask = {}
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = self.prepare_sparse_mask(video_mask, prompt_mask, sparse_n)

        # 2. Blocks
        frames = torch.tensor(frames)
        height = torch.tensor(height)
        width = torch.tensor(width)

        for i, block in enumerate(self.videodit_sparse_blocks):
            if i > 1 and i < 30:
                video_mask, prompt_mask = sparse_mask[block.self_atten.sparse_n][block.self_atten.sparse_group]
            else:
                video_mask, prompt_mask = sparse_mask[1][block.self_atten.sparse_group]

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                latents = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    latents,
                    video_mask,
                    prompt,
                    prompt_mask,
                    timestep,
                    frames,
                    height,
                    width
                )
            else:
                latents = block(
                            latents,
                            video_mask=video_mask,
                            prompt=prompt,
                            prompt_mask=prompt_mask,
                            timestep=timestep,
                            frames=frames,
                            height=height,
                            width=width,
                        )

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        latents = rearrange(latents, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            latents=latents,
            timestep=timestep,
            embedded_timestep=embedded_timestep,
            num_frames=frames,
            height=height,
            width=width,
        )  # b c t h w

        if self.training and mpu.get_context_parallel_world_size() > 1:
            output = gather_forward_split_backward(output, mpu.get_context_parallel_group(), dim=2,
                                                        grad_scale='up')

        return output

    def _get_block(self, layer_number):
        return self.videodit_sparse_blocks[layer_number]

    def _checkpointed_forward(
            self,
            sparse_mask,
            latents,
            video_mask,
            prompt,
            prompt_mask,
            timestep,
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
                    video_mask,
                    prompt,
                    prompt_mask,
                    timestep,
                    frames,
                    height,
                    width
                )
                layer_num += self.recompute_num_layers
        elif self.recompute_method == "block":
            for layer_num in range(self.videodit_sparse_blocks):
                block = self._get_block(layer_num)
                if layer_num > 1 and layer_num < 30:
                    video_mask, prompt_mask = sparse_mask[block.self_atten.sparse_n][block.self_atten.sparse_group]
                else:
                    video_mask, prompt_mask = sparse_mask[1][block.self_atten.sparse_group]
                if layer_num < self.recompute_num_layers:
                    latents = tensor_parallel.checkpoint(
                        custom(layer_num, layer_num + 1),
                        self.distribute_saved_activations,
                        latents,
                        video_mask,
                        prompt,
                        prompt_mask,
                        timestep,
                        frames,
                        height,
                        width
                    )
                else:
                    # block = self._get_block(layer_num)
                    latents = block(
                        latents,
                        video_mask,
                        prompt,
                        prompt_mask,
                        timestep,
                        frames,
                        height,
                        width
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

    def _operate_on_patched_inputs(self, latents, prompt, timestep, batch_size):

        latents = self.pos_embed(latents.to(self.dtype))

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        prompt = self.caption_projection(prompt)  # b, 1, l, d or b, 1, l, d
        if prompt.shape[1] != 1:
            raise ValueError("prompt's shape mismatched")
        prompt = rearrange(prompt, 'b 1 l d -> (b 1) l d')

        return latents, prompt, timestep, embedded_timestep

    def _get_output_for_patched_inputs(
            self, latents, timestep, embedded_timestep, num_frames, height, width
    ):
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        latents = self.norm_out(latents)
        # Modulation
        latents = latents * (1 + scale) + shift
        latents = self.proj_out(latents)
        latents = latents.squeeze(1)

        # unpatchify
        latents = latents.reshape(
            shape=(
            -1, num_frames, height, width, self.patch_size_t, self.patch_size, self.patch_size,
            self.out_channels)
        )
        latents = torch.einsum("nthwopqc->nctohpwq", latents)
        output = latents.reshape(
            shape=(-1, self.out_channels,
                   num_frames * self.patch_size_t, height * self.patch_size,
                   width * self.patch_size)
        )
        return output


class VideoDiTSparseBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            attention_bias: bool = False,
            attention_out_bias: bool = True,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            interpolation_scale: Tuple[int] = (1, 1, 1),
            sparse1d: bool = False,
            sparse_n: int = 2,
            sparse_group: bool = False,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

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
            latents: torch.FloatTensor,
            video_mask: Optional[torch.FloatTensor] = None,
            prompt: Optional[torch.FloatTensor] = None,
            prompt_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            frames: int = None,
            height: int = None,
            width: int = None,
    ) -> torch.FloatTensor:
        # 1. Self-Attention
        batch_size = latents.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
        ).chunk(6, dim=0)
        norm_latents = self.norm1(latents)
        norm_latents = norm_latents * (1 + scale_msa) + shift_msa
        attn_output = self.self_atten(
            query=norm_latents,
            key=None,
            mask=video_mask,
            frames=frames,
            height=height,
            width=width,
        )
        attn_output = gate_msa * attn_output
        latents = attn_output + latents

        # 2. Cross-Attention
        norm_latents = latents
        attn_output = self.cross_atten(
            query=norm_latents,
            key=prompt,
            mask=prompt_mask,
            frames=frames,
            height=height,
            width=width,
        )
        latents = attn_output + latents

        # 3. Feed-forward
        norm_latents = self.norm2(latents)
        norm_latents = norm_latents * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_latents)
        ff_output = gate_mlp * ff_output
        latents = ff_output + latents
        return latents


