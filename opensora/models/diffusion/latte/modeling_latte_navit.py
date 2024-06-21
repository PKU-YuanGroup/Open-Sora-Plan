import torch

import os
import json

from dataclasses import dataclass
from einops import rearrange, repeat
from diffusers.utils import USE_PEFT_BACKEND, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from opensora.models.diffusion.utils.pos_embed import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed, PositionGetter1D, PositionGetter2D
from opensora.models.diffusion.latte.modules import BasicTransformerBlock, BasicTransformerBlock_, AdaLayerNormSingle, CaptionProjection

def video_grouping(
    videos, patch_size, max_token_lim=4096, token_dropout_rate=0.0
):
    """
    Use greedy algorithms to group videos into groups with num_token less than max_token_lim

    Args:
        videos: List of tensor with shape (F, C, H, W), F is the number of frames,
        C is the number of channels, and H and W are the height and width respectively.

    Returns:
        groups: A list of lists containing torch.Tensors, each having shape (F, C, H, W).
    """

    groups, video_ids = [[]], [[]]

    # greedy algorithm is a bit complex to implement so we use the naive
    # sequential grouping for now.
    # TODO: implement greedy algorithm
    seq_len = 0
    for idx, video in enumerate(videos):
        assert isinstance(video, torch.Tensor), "video must be torch.Tensor"
        assert video.ndim == 4, "video must be 4d tensor"

        video_h, video_w = video.shape[-2:]
        assert (video_w % patch_size) == 0 and (
            video_h % patch_size
        ) == 0, f"video width and height must be divisible by patch size {patch_size}"
        patch_w, patch_h = video_w // patch_size, video_h // patch_size

        token_len = int(patch_w * patch_h * (1 - token_dropout_rate))
        assert (
            token_len <= max_token_lim
        ), f"token length {token_len} exceeds max token length {max_token_lim}"
        if seq_len + token_len <= max_token_lim:
            groups[-1].append(video)
            video_ids[-1].append(idx)
            seq_len += token_len
        else:
            groups.append([video])
            video_ids.append([idx])
            seq_len = token_len

    return groups, video_ids


def pack_timestep_as(timestep, video_ids, num_patches):
    # timestep: (B, D)
    batched_output = []
    for group, num_patch in zip(video_ids, num_patches):
        output = torch.empty(
            (0,),
            device=timestep.device,
            dtype=timestep.dtype,
        )
        for sample_idx, t in zip(group, num_patch):
            # sample: (D,)
            sample = timestep[sample_idx]
            # (T, D)
            sample = repeat(sample.unsqueeze(0), "1 n -> (1 t) n", t=t)
            output = torch.cat([output, sample], dim=0)
        batched_output.append(output)

    # (b, T, D)
    batched_output = nn.utils.rnn.pad_sequence(
        batched_output, batch_first=True
    )
    return batched_output


def pack_text_as(encoder_hidden_states, encoder_attention_mask, video_ids):
    # encoder_hidden_states: (B, L, D)
    # encoder_attention_mask: (B, L)
    encoder_attention_mask = encoder_attention_mask.bool()
    batched_output = []
    batched_idx = []
    for group in video_ids:
        output = torch.empty(
            (0,),
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        text_idx = torch.empty(
            (0,),
            device=encoder_hidden_states.device,
            dtype=torch.long,
        )
        for idx, sample_idx in enumerate(group):
            # (L, D)
            sample = encoder_hidden_states[sample_idx]
            # (L,)
            padding_mask = encoder_attention_mask[sample_idx]
            # (l, D)   l <= L
            # discard padded tokens before packing
            non_padding_sample = sample[padding_mask]
            output = torch.cat([output, non_padding_sample], dim=0)
            text_idx = torch.cat(
                (
                    text_idx,
                    torch.full(
                        (non_padding_sample.shape[0],),
                        idx,
                        device=encoder_hidden_states.device,
                        dtype=torch.long,
                    ),  # (l,)
                )
            )
        batched_output.append(output)
        batched_idx.append(text_idx)

    # (b, L', D)   L' = max(sum(l))
    batched_output = nn.utils.rnn.pad_sequence(
        batched_output, batch_first=True
    )
    # (b, L')
    batched_idx = nn.utils.rnn.pad_sequence(
        batched_idx, batch_first=True, padding_value=-2
    )
    return batched_output, batched_idx

def pack_image_joint_text_as(encoder_hidden_states, encoder_attention_mask, video_ids):
    # encoder_hidden_states: (B, F, L, D)
    # encoder_attention_mask: (B, F, L)
    # 0: valid, -1: padding
    encoder_attention_mask = encoder_attention_mask.bool()
    batched_output = []
    batched_idx = []
    frame = encoder_hidden_states.shape[1]
    for group in video_ids:
        output = torch.empty(
            (0,),
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        text_idx = torch.empty(
            (0,),
            device=encoder_hidden_states.device,
            dtype=torch.long,
        )
        for idx, sample_idx in enumerate(group):
            # (F, L, D)
            sample = encoder_hidden_states[sample_idx]
            # (F, L)
            padding_mask = ~encoder_attention_mask[sample_idx]
            # (F, L,)
            cur_text_idx = torch.full(
                        tuple(padding_mask.shape),
                        idx,
                        device=encoder_hidden_states.device,
                        dtype=torch.long,
                    ).masked_fill(padding_mask, -2)
            output = torch.cat([output, sample], dim=1)
            text_idx = torch.cat(
                (
                    text_idx,
                    cur_text_idx,
                ),
                dim=1,
            )
        batched_output.append(rearrange(output, "f l d -> l (f d)"))
        batched_idx.append(text_idx.transpose(-1,-2))
    
    # # (b, F, L', D)
    # batched_output = torch.stack(batched_output)
    # # (b, F, L')
    # batched_idx = torch.stack(batched_idx)

    # (b, F, L', D)
    batched_output = nn.utils.rnn.pad_sequence(
        batched_output, batch_first=True
    )
    batched_output = rearrange(batched_output, "b l (f d) -> b f l d", f=frame)
    # (b, F, L')
    batched_idx = nn.utils.rnn.pad_sequence(
        batched_idx, batch_first=True, padding_value=-2
    )
    batched_idx = batched_idx.transpose(-1,-2)
    return batched_output, batched_idx


def pack_target_as(targets, video_ids, patch_size, token_kept_ids=None):
    # targets: List of (C, F, H, W)
    device, dtype = targets[0].device, targets[0].dtype
    num_frame = targets[0].shape[1]
    batched_output = []
    for group_id, group in enumerate(video_ids):
        output = torch.empty(
            (0,),
            device=device,
            dtype=dtype,
        )
        for idx, sample_idx in enumerate(group):
            # (C, F, H, W)
            target = targets[sample_idx]
            height, width = target.shape[-2:]
            num_patch_h, num_patch_w = (
                height // patch_size,
                width // patch_size,
            )
            # (T, F*p*p*c)
            target = rearrange(
                target,
                "c f (h p) (w q)-> (h w) (f p q c)",
                h=num_patch_h,
                w=num_patch_w,
                p=patch_size,
                q=patch_size,
            )
            if token_kept_ids is not None:
                target = target[token_kept_ids[group_id][idx]]
            output = torch.cat([output, target], dim=0)
        batched_output.append(output)

    # (b, T, F*p*p*c)
    batched_output = nn.utils.rnn.pad_sequence(
        batched_output, batch_first=True
    )
    # (b, F, T, D)
    batched_output = rearrange(
        batched_output, "b t (f d) -> b f t d", f=num_frame
    )
    return batched_output


def mask_to_bias(attention_mask, dtype):
    """
    convert attention_mask to a bias
    expects mask of shape:
      [batch, query_tokens, key_tokens]
    this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
      [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
      [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    assume that mask is expressed as:
      (True = keep,      False = discard)
    convert mask into a bias that can be added to attention scores:
          (keep = +0,     discard = -10000.0)
    """
    return (1 - attention_mask.to(dtype)) * -10000.0


class ExamplePacking(nn.Module):
    """3D video to Patch Embedding"""

    def __init__(
        self,
        base_height=512,  # 4096 -> vae x 8 down -> 512
        base_width=512,
        patch_size=2,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        bias=True,
        interpolation_scale=1,
        max_token_lim=1024,
        token_dropout_rate=0.0,
    ):
        super().__init__()
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, elementwise_affine=False, eps=1e-6
            )
        else:
            self.norm = None

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.base_height, self.base_width = (base_height, base_width)
        self.base_num_patches = (base_height // patch_size) * (
            base_width // patch_size
        )
        self.base_size = base_height // patch_size
        self.interpolation_scale = interpolation_scale
        self.max_token_lim = max_token_lim
        self.token_dropout_rate = token_dropout_rate
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim,
            int(self.base_num_patches**0.5),
            base_size=self.base_size,
            interpolation_scale=self.interpolation_scale,
        )
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed),
            persistent=False,
        )

    def forward(self, latent, dtype):
        """
        Pack latent from different batches to one sequence and proj to patch embed.
        Args:
            latent: List of list of tensor with shape (F, C, H, W), where F is the number of frames,
            C is the number of channels, and H and W are the height and width respectively.

        Returns:
            output: Output tensor with shape (b, F, T, D)
        """

        num_frame = latent[0][0].shape[0]
        device = latent[0][0].device
        video_groups, video_ids = video_grouping(
            latent,
            self.patch_size,
            max_token_lim=self.max_token_lim,
            token_dropout_rate=self.token_dropout_rate,
        )

        batched_video = []
        batched_pos = []
        batched_idx = []

        batched_len = []
        max_len = 0
        # group_size = []  # number of videos of each group
        num_patches = []  # number of patches of each video in each seq
        token_kept_ids = [] # token left after random dropping

        for group in video_groups:
            # group_size.append(len(group))
            num_patches.append([])
            token_kept_ids.append([])
            video_seq = torch.empty(
                (num_frame, 0, self.embed_dim),
                device=device,
                dtype=dtype,
            )
            video_pos = torch.empty(
                (0, self.embed_dim), device=device, dtype=dtype
            )
            video_idx = torch.empty((0,), device=device, dtype=torch.long)

            for idx, video in enumerate(group):
                # (F, C, H, W)
                video = video.to(dtype)
                video = rearrange(video, "c f h w -> f c h w").contiguous()

                height, width = video.shape[-2:]
                num_patch_h, num_patch_w = (
                    height // self.patch_size,
                    width // self.patch_size,
                )

                # (F, C, H, W) -> (F, D, H//P, W//P)
                seq = self.proj(video)
                # (F, D, H//P, W//P) -> (F, T, D)
                seq = seq.flatten(2).transpose(1, 2)

                # Interpolate positional embeddings if needed.
                # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
                if height != self.base_height or width != self.base_width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.embed_dim,
                        grid_size=(num_patch_h, num_patch_w),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                    )  # (T, D)
                    pos_embed = torch.from_numpy(pos_embed).to(device)

                else:
                    pos_embed = self.pos_embed

                assert (
                    pos_embed.shape[0] == seq.shape[1]
                ), "pos_embed and sequence token length mismatch"

                if self.token_dropout_rate > 0:
                    selected_len = int(
                        seq.shape[1] * (1 - self.token_dropout_rate)
                    )
                    select_indices = torch.randperm(
                        seq.shape[1], device=device
                    )[:selected_len]
                    seq = seq[:, select_indices]
                    pos_embed = pos_embed[select_indices]
                    token_kept_ids[-1].append(select_indices)

                num_patches[-1].append(seq.shape[1])

                video_seq = torch.cat([video_seq, seq], dim=1)
                video_pos = torch.cat([video_pos, pos_embed], dim=0)
                video_idx = torch.cat(
                    (
                        video_idx,
                        torch.full(
                            (seq.shape[1],),
                            idx,
                            device=device,
                            dtype=torch.long,
                        ),  # (T,)
                    )
                )
            batched_video.append(rearrange(video_seq, "f t d -> t (f d)"))
            batched_pos.append(video_pos)
            batched_idx.append(video_idx)
            # [t1, t2, t3, ...]
            batched_len.append(video_seq.shape[1])
            if video_seq.shape[1] > max_len:
                max_len = video_seq.shape[1]

        # (b, T, (F * D))
        batched_video = nn.utils.rnn.pad_sequence(
            batched_video, batch_first=True
        )
        # (b, F, T, D)
        batched_video = rearrange(
            batched_video, "b t (f d) -> b f t d", f=num_frame
        )

        # (b, T, D)
        batched_pos = nn.utils.rnn.pad_sequence(batched_pos, batch_first=True)
        # (b, 1, T, D)
        batched_pos = batched_pos.unsqueeze(1)

        # (b, T)
        batched_idx = nn.utils.rnn.pad_sequence(
            batched_idx, batch_first=True, padding_value=-1
        )

        if self.layer_norm:
            batched_video = self.norm(batched_video)

        if not self.token_dropout_rate > 0:
            token_kept_ids = None

        return (
            (batched_video + batched_pos).to(dtype),
            batched_idx,
            video_ids,
            num_patches,
            token_kept_ids,
        )

class NaViTLatteT2V(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

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

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            patch_size_t: int = 1,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            video_length: int = 16,
            attention_mode: str = 'flash', 
            use_rope: bool = False, 
            rope_scaling_type: str = 'linear', 
            compress_kv_factor: int = 1, 
            interpolation_scale_1d: float = None, 
            max_token_lim: int = 1024,
            token_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length
        self.use_rope = use_rope
        self.compress_kv_factor = compress_kv_factor
        self.num_layers = num_layers

        assert self.compress_kv_factor == 1 and not use_rope, "NaViT currently does not support compressing kv or using rope"

        conv_cls = nn.Conv2d if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        # self.is_input_patches = in_channels is not None and patch_size is not None
        self.is_input_patches = True

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Define input layers
        assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

        self.height = sample_size[0]
        self.width = sample_size[1]

        self.patch_size = patch_size
        interpolation_scale_2d = self.config.sample_size[0] // 64  # => 64 (= 512 pixart) has interpolation scale 1
        interpolation_scale_2d = max(interpolation_scale_2d, 1)
        self.pos_embed = ExamplePacking(
            # position encoding will interpolate to sample_size which is determined by args.max_image_size
            base_height=self.height,
            base_width=self.width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale_2d,
            max_token_lim=max_token_lim,
            token_dropout_rate=token_dropout_rate,
        )

        
        # define temporal positional embedding
        if interpolation_scale_1d is None:
            if self.config.video_length % 2 == 1:
                interpolation_scale_1d = (self.config.video_length - 1) // 16  # => 16 (= 16 Latte) has interpolation scale 1
            else:
                interpolation_scale_1d = self.config.video_length // 16  # => 16 (= 16 Latte) has interpolation scale 1
        # interpolation_scale_1d = self.config.video_length // 5  # 
        interpolation_scale_1d = max(interpolation_scale_1d, 1)
        temp_pos_embed = get_1d_sincos_pos_embed(inner_dim, video_length, interpolation_scale=interpolation_scale_1d)  # 1152 hidden size
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

        rope_scaling = None
        if self.use_rope:
            self.position_getter_2d = PositionGetter2D()
            self.position_getter_1d = PositionGetter1D()
            rope_scaling = dict(type=rope_scaling_type, factor_2d=interpolation_scale_2d, factor_1d=interpolation_scale_1d)

        # 3. Define transformers blocks, spatial attention
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    attention_mode=attention_mode, 
                    use_rope=use_rope, 
                    rope_scaling=rope_scaling, 
                    compress_kv_factor=(compress_kv_factor, compress_kv_factor) if d >= num_layers // 2 and compress_kv_factor != 1 else None, # follow pixart-sigma, apply in second-half layers
                )
                for d in range(num_layers)
            ]
        )

        # Define temporal transformers blocks
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock_(  # one attention
                    inner_dim,
                    num_attention_heads,  # num_attention_heads
                    attention_head_dim,  # attention_head_dim 72
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=False,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    attention_mode=attention_mode, 
                    use_rope=use_rope, 
                    rope_scaling=rope_scaling, 
                    compress_kv_factor=(compress_kv_factor, ) if d >= num_layers // 2 and compress_kv_factor != 1 else None, # follow pixart-sigma, apply in second-half layers
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        elif self.is_input_patches and norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            # self.use_additional_conditions = self.config.sample_size[0] == 128  # False, 128 -> 1024
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def make_position(self, b, t, use_image_num, h, w, device):
        pos_hw = self.position_getter_2d(b*(t+use_image_num), h, w, device)  # fake_b = b*(t+use_image_num)
        pos_t = self.position_getter_1d(b*h*w, t, device)  # fake_b = b*h*w
        return pos_hw, pos_t

    def make_attn_mask(self, attention_mask, frame, dtype):
        attention_mask = rearrange(attention_mask, 'b t h w -> (b t) 1 (h w)') 
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #   (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(dtype)) * -10000.0
        attention_mask = attention_mask.to(self.dtype)
        return attention_mask

    def vae_to_diff_mask(self, attention_mask, use_image_num):
        dtype = attention_mask.dtype
        # b, t+use_image_num, h, w, assume t as channel
        # this version do not use 3d patch embedding
        attention_mask = F.max_pool2d(attention_mask, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        attention_mask = attention_mask.bool().to(dtype)
        return attention_mask

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
            use_image_num: int = 0,
            enable_temporal_attentions: bool = True,
            return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states: List of tensors. B(C, 1+F+num_img, H, W).
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
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        input_batch_size = len(hidden_states)
        # (b, F, T, D)
        (hidden_states, batched_idx_video, video_ids, num_patches, token_kept_ids) = (
            self.pos_embed(hidden_states, dtype=self.dtype)
        )  # alrady add positional embeddings

        packed_batch_size, frame, seq_len, _ = hidden_states.shape
        frame = frame - use_image_num  # 20-4=16s

        # (b*F, T, D)
        hidden_states = rearrange(hidden_states, "b f t d -> (b f) t d")

        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        # (B, D * 6), (B, D)
        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs,
            batch_size=input_batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        # (b, T, D * 6)
        timestep = pack_timestep_as(timestep, video_ids, num_patches)
        embedded_timestep = pack_timestep_as(
            embedded_timestep, video_ids, num_patches
        )
        assert timestep.shape[1] == hidden_states.shape[1]
        assert embedded_timestep.shape[1] == hidden_states.shape[1]
        

        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = repeat(encoder_attention_mask, 'b l -> b f l', f=frame).contiguous()
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(encoder_attention_mask_video, 'b 1 l -> b (1 f) l',
                                                  f=frame).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)

        # 2. Blocks
        if self.caption_projection is not None:
            # (B, L, D) or (B, 1+num_image, L, D)
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  # 3 120 1152

            # (B, F, L, D)
            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(encoder_hidden_states_video, 'b 1 t d -> b (1 f) t d', f=frame).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
            else:
                encoder_hidden_states_spatial = repeat(encoder_hidden_states, 'b t d -> b f t d', f=frame).contiguous()

        # (b, F, L', D), (b, F, L')
        encoder_hidden_states, batched_idx_text = pack_image_joint_text_as(
            encoder_hidden_states, encoder_attention_mask, video_ids
        )
        encoder_hidden_states_spatial = rearrange(encoder_hidden_states, 'b f t d -> (b f) t d').contiguous()

        # compute self-attn mask
        assert attention_mask is None, "NaViT does not support attention_mask!"
        # (b, T, T) True for keep and False for discard
        attention_mask = batched_idx_video.unsqueeze(
            -1
        ) == batched_idx_video.unsqueeze(1)
        # (b, T) True for keep and False for discard
        padding_mask_1d = batched_idx_video >= 0
        # (b, T, T)
        padding_mask = padding_mask_1d.unsqueeze(
            -1
        ) & padding_mask_1d.unsqueeze(1)
        # (b, T, T)
        attention_mask = attention_mask & padding_mask

        # compute cross-attn mask with text condition
        # (b, F, T, L) True for keep and False for discard
        encoder_attention_mask = batched_idx_video.unsqueeze(1).unsqueeze(
            -1
        ) == batched_idx_text.unsqueeze(2)

        assert (
            encoder_attention_mask.shape[-1]
            == encoder_hidden_states_spatial.shape[-2]
        )

        # convert bool mask to bias
        attention_mask = mask_to_bias(attention_mask, hidden_states.dtype)
        encoder_attention_mask = mask_to_bias(
            encoder_attention_mask, hidden_states.dtype
        )

        attention_mask = repeat(
            attention_mask, "b t l -> (b f) t l", f=frame + use_image_num
        )
        encoder_attention_mask = rearrange(
            encoder_attention_mask, "b f t l -> (b f) t l"
        )

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(
            timestep, "b t d -> (b f) t d", f=frame + use_image_num
        ).contiguous()
        timestep_temp = rearrange(
            timestep, "b t d -> (b t) d", t=seq_len
        ).contiguous()
        

        for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    attention_mask, 
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )

                if enable_temporal_attentions:
                    hidden_states = rearrange(hidden_states, '(b f) t d -> (b t) f d', b=packed_batch_size).contiguous()

                    if use_image_num != 0:  # image-video joitn training
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states_video = hidden_states_video + self.temp_pos_embed

                        hidden_states_video = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            use_reentrant=False,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=packed_batch_size).contiguous()

                    else:
                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states = hidden_states + self.temp_pos_embed

                        hidden_states = torch.utils.checkpoint.checkpoint(
                            temp_block,
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                            use_reentrant=False,
                        )

                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=packed_batch_size).contiguous()
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    attention_mask, 
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                )

                if enable_temporal_attentions:
                    # b c f h w, f = 16 + 4
                    hidden_states = rearrange(hidden_states, '(b f) t d -> (b t) f d', b=packed_batch_size).contiguous()

                    if use_image_num != 0 and self.training:
                        hidden_states_video = hidden_states[:, :frame, ...]
                        hidden_states_image = hidden_states[:, frame:, ...]

                        # if i == 0 and not self.use_rope:
                        #     hidden_states_video = hidden_states_video + self.temp_pos_embed

                        hidden_states_video = temp_block(
                            hidden_states_video,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                        )

                        hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=packed_batch_size).contiguous()

                    else:
                        # if i == 0 and not self.use_rope:
                        if i == 0:
                            hidden_states = hidden_states + self.temp_pos_embed

                        hidden_states = temp_block(
                            hidden_states,
                            None,  # attention_mask
                            None,  # encoder_hidden_states
                            None,  # encoder_attention_mask
                            timestep_temp,
                            cross_attention_kwargs,
                            class_labels,
                        )

                        hidden_states = rearrange(hidden_states, '(b t) f d -> (b f) t d',
                                                  b=packed_batch_size).contiguous()



        embedded_timestep = repeat(
            embedded_timestep,
            "b t d -> (b f) t d",
            f=frame + use_image_num,
        ).contiguous()
        params = (
            self.scale_shift_table[None, None]  # [1, 1, 2, D]
            + embedded_timestep[:, :, None]  # [b*F, T, 1, D]
        ).chunk(2, dim=-2)
        shift, scale = [param.squeeze(-2) for param in params]
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift

        # [b*F, T, p*p*c]
        hidden_states = self.proj_out(hidden_states)
        # [b, F, T, p*p*c]
        hidden_states = rearrange(
            hidden_states, "(b f) t d -> b f t d", b=packed_batch_size
        )
        # padding_mask_1d: (b, T) True for keep and False for discard
        # make sure padded token filled with 0
        # (b, 1, T, 1)
        padding_mask = padding_mask_1d[:, None, :, None]

        hidden_states = hidden_states.masked_fill(~padding_mask, 0)

        return hidden_states, video_ids, token_kept_ids

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model = cls.from_config(config, **kwargs)
        return model

# depth = num_layers * 2
def NaViTLatteT2V_XL_122(**kwargs):
    return NaViTLatteT2V(num_layers=28, attention_head_dim=72, num_attention_heads=16, patch_size_t=1, patch_size=2,
                    norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)
def NaViTLatteT2V_D64_XL_122(**kwargs):
    return NaViTLatteT2V(num_layers=28, attention_head_dim=64, num_attention_heads=18, patch_size_t=1, patch_size=2,
                    norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)

Latte_navit_models = {
    "NaViTLatteT2V-XL/122": NaViTLatteT2V_XL_122,
    "NaViTLatteT2V-D64-XL/122": NaViTLatteT2V_D64_XL_122,
}
