from selectors import EpollSelector
from typing import Any, Dict, Optional, Tuple, List

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.utils import is_torch_version
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.legacy.model.layer_norm import LayerNorm
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.utils import print_rank_0 as print

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.embeddings import PatchEmbed2D, RoPE3D, PositionGetter3D, apply_rotary_emb
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.attention import MultiHeadSparseMMAttentionSBH
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward

from mindspeed_mm.models.predictor.dits.modules import CombinedTimestepTextProjEmbeddings, AdaNorm, OpenSoraNormZero

selective_recom = True
recom_ffn_layers = 32

def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)
    return custom_forward

def zero_initialized_skip_connection(module_cls):
    if not issubclass(module_cls, nn.Linear):
        raise TypeError(f"Expected module_cls to be nn.Linear, but got {module_cls.__name__}.")
    def zero_init(*args, **kwargs):
        module = module_cls(*args, **kwargs)
        in_features = module.in_features
        out_features = module.out_features
        if in_features != 2 * out_features:
            raise ValueError("Expected in_features to be twice out_features, "
                             f"but got in_features={in_features} and out_features={out_features}.")

        module.weight.data[:, :out_features] = torch.eye(out_features, dtype=module.weight.dtype)
        module.weight.data[:, out_features:] = 0.0
        if module.bias is not None:
            module.bias.data.fill_(0.0)
        return module
    return zero_init

def maybe_clamp_tensor(x, max_value=65504.0, min_value=-65504.0, training=True):
    if not training and x.dtype == torch.float16:
        x.nan_to_num_(posinf=max_value, neginf=min_value).clamp_(min_value, max_value)
    return x

class SparseUMMDiT(MultiModalModule):
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
        num_layers: List[int] = [2, 4, 8, 4, 2], 
        sparse_n: List[int] = [1, 4, 16, 4, 1], 
        double_ff: bool = False,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        pooled_projection_dim: int = 1024, 
        timestep_embed_dim: int = 512,
        norm_cls: str = 'layer_norm',
        skip_connection: bool = False,
        explicit_uniform_rope: bool = False, 
        skip_connection_zero_init: bool = True,
        **kwargs
    ):
        super().__init__(config=None)
        args = get_args()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sequence_parallel = args.sequence_parallel
        print(f'sequence_parallel: {args.sequence_parallel}')
        self.recompute_num_layers = args.recompute_num_layers
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        hidden_size = num_heads * head_dim
        self.num_layers = num_layers
        self.sparse_n = sparse_n
        self.patch_size_t = patch_size_t
        self.patch_size = patch_size
        self.skip_connection = skip_connection
        self.skip_connection_zero_init = skip_connection_zero_init

        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = LayerNorm

        if len(num_layers) != len(sparse_n):
            raise ValueError("num_layers and sparse_n must have the same length")
        if len(num_layers) % 2 != 1:
            raise ValueError("num_layers must have odd length")
        if any([i % 2 != 0 for i in num_layers]):
            raise ValueError("num_layers must have even numbers")

        if not sparse1d:
            self.sparse_n = [1] * len(num_layers)

        interpolation_scale_thw = (interpolation_scale_t, interpolation_scale_h, interpolation_scale_w)

        # 1. patch embedding
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # 2. time embedding and pooled text embedding
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            timestep_embed_dim=timestep_embed_dim, 
            embedding_dim=timestep_embed_dim, 
            pooled_projection_dim=pooled_projection_dim
        )

        # 3. anthor text embedding
        self.caption_projection = nn.Linear(caption_channels, hidden_size)

        # 4. rope
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
        self.position_getter = PositionGetter3D(
            sample_size_t, sample_size_h, sample_size_w, explicit_uniform_rope, atten_layout="SBH"
        )

        # forward transformer blocks
        self.transformer_blocks = []
        self.skip_norm_linear = []
        self.skip_norm_linear_enc = []

        for idx, (num_layer, sparse_n) in enumerate(zip(self.num_layers, self.sparse_n)):
            is_last_stage = idx == len(num_layers) - 1
            if self.skip_connection and idx > len(num_layers) // 2:
                skip_connection_linear = zero_initialized_skip_connection(nn.Linear) if self.skip_connection_zero_init else nn.Linear
                self.skip_norm_linear.append(
                    nn.Sequential(
                        self.norm_cls(
                            hidden_size * 2,
                            eps=norm_eps,
                            sequence_parallel=self.sequence_parallel,
                        ) if not self.skip_connection_zero_init else nn.Identity(),
                        skip_connection_linear(hidden_size * 2, hidden_size),
                    )
                )
                self.skip_norm_linear_enc.append(
                    nn.Sequential(
                        self.norm_cls(
                            hidden_size * 2,
                            eps=norm_eps,
                            sequence_parallel=self.sequence_parallel,
                        ) if not self.skip_connection_zero_init else nn.Identity(),
                        skip_connection_linear(hidden_size * 2, hidden_size),
                    )
                )
            stage_blocks = nn.ModuleList(
                [
                    SparseMMDiTBlock(
                        dim=hidden_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        timestep_embed_dim=timestep_embed_dim,
                        dropout=dropout,
                        activation_fn=activation_fn,
                        attention_bias=attention_bias,
                        norm_elementwise_affine=norm_elementwise_affine,
                        norm_eps=norm_eps,
                        interpolation_scale_thw=interpolation_scale_thw,
                        double_ff=double_ff,
                        sparse1d=sparse1d if sparse_n > 1 else False,
                        sparse_n=sparse_n,
                        sparse_group=i % 2 == 1 if sparse_n > 1 else False,
                        context_pre_only=is_last_stage and (i == num_layer - 1),
                        norm_cls=norm_cls,
                    )
                    for i in range(num_layer)
                ]
            )
            self.transformer_blocks.append(stage_blocks)
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)

        if self.skip_connection:
            self.skip_norm_linear = nn.ModuleList(self.skip_norm_linear)
            self.skip_norm_linear_enc = nn.ModuleList(self.skip_norm_linear_enc)

        self.norm_final = self.norm_cls(
            hidden_size, eps=norm_eps, sequence_parallel=self.sequence_parallel,
        )

        self.norm_out = AdaNorm(
            embedding_dim=timestep_embed_dim,
            output_dim=hidden_size * 2,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            norm_cls=norm_cls,
        )

        self.proj_out = nn.Linear(
            hidden_size, patch_size_t * patch_size * patch_size * out_channels
        )

        # set label "sequence_parallel", for all_reduce the grad
        modules = [self.norm_final]
        if self.skip_connection:
            modules += [self.skip_norm_linear, self.skip_norm_linear_enc]
        for module in modules:
            for name, param in module.named_parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)

    def prepare_sparse_mask(self, attention_mask, encoder_attention_mask, sparse_n):
        attention_mask = attention_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        l = attention_mask.shape[-1]
        if l % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - l % (sparse_n * sparse_n)

        attention_mask_sparse = F.pad(attention_mask, (0, pad_len, 0, 0), value=-10000.0)
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
 
        # concat mask at sequence dim
        attention_mask_sparse_1d = torch.cat([attention_mask_sparse_1d, encoder_attention_mask_sparse], dim=-1)
        attention_mask_sparse_1d_group = torch.cat([attention_mask_sparse_1d_group, encoder_attention_mask_sparse], dim=-1)

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

        return {
            False: attention_mask_sparse_1d,
            True: attention_mask_sparse_1d_group
        }


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, c, frames, height, width = hidden_states.shape

        # print(f"model forward, hidden_states: {hidden_states.shape}, timestep: {timestep.shape}, pooled_projections: {pooled_projections.shape}, encoder_hidden_states: {encoder_hidden_states.shape}, attention_mask: {attention_mask.shape}, encoder_attention_mask: {encoder_attention_mask.shape}")
        encoder_attention_mask = encoder_attention_mask.view(batch_size, -1, encoder_attention_mask.shape[-1])
        if self.training and mpu.get_context_parallel_world_size() > 1:
            frames //= mpu.get_context_parallel_world_size()
            hidden_states = split_forward_gather_backward(hidden_states, mpu.get_context_parallel_group(), dim=2,
                                                    grad_scale='down')
            encoder_hidden_states = split_forward_gather_backward(encoder_hidden_states, mpu.get_context_parallel_group(),
                                                   dim=2, grad_scale='down')

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask.to(self.dtype)

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = F.max_pool3d(
                attention_mask,
                kernel_size=(self.patch_size_t, self.patch_size, self.patch_size),
                stride=(self.patch_size_t, self.patch_size, self.patch_size)
            )
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)')
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:
            # b, 1, l
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0

        # 1. Input
        frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
        height, width = height // self.patch_size, width // self.patch_size

        hidden_states, encoder_hidden_states, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, pooled_projections
        )

        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()

        self.sparse_mask = {}

        for sparse_n in list(set(self.sparse_n)):
            self.sparse_mask[sparse_n] = self.prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n)

        pos_thw = self.position_getter(
            batch_size, t=frames * mpu.get_context_parallel_world_size(), h=height, w=width,
            device=hidden_states.device, training=self.training
        )
        video_rotary_emb = self.rope(self.head_dim, pos_thw, hidden_states.device)

        if self.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)
            encoder_hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(encoder_hidden_states)

        hidden_states, encoder_hidden_states, skip_connections = self._operate_on_enc(
            hidden_states, encoder_hidden_states, embedded_timestep, frames, height, width, video_rotary_emb
        )

        hidden_states, encoder_hidden_states = self._operate_on_mid(
            hidden_states, encoder_hidden_states, embedded_timestep, frames, height, width, video_rotary_emb
        )

        hidden_states, encoder_hidden_states = self._operate_on_dec(
            hidden_states, skip_connections, encoder_hidden_states, embedded_timestep, frames, height, width, video_rotary_emb
        )

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states, embedded_timestep, frames, height, width
        )  # b c t h w

        if self.training and mpu.get_context_parallel_world_size() > 1:
            output = gather_forward_split_backward(output, mpu.get_context_parallel_group(), dim=2,
                                                        grad_scale='up')

        return output

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def _operate_on_enc(
        self, hidden_states, encoder_hidden_states,
        embedded_timestep, frames, height, width, video_rotary_emb
    ):
        layer_idx = 0
        skip_connections = []
        for idx, stage_block in enumerate(self.transformer_blocks[:len(self.num_layers) // 2]):
            for idx_, block in enumerate(stage_block):
                layer_idx += 1
                attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
                hidden_states, encoder_hidden_states = self.block_forward(
                    block=block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    frames=frames,
                    height=height,
                    width=width,
                    video_rotary_emb=video_rotary_emb,
                    layer_idx=layer_idx
                )
            if self.skip_connection:
                skip_connections.append([hidden_states, encoder_hidden_states])
        return hidden_states, encoder_hidden_states, skip_connections
    
    def _operate_on_mid(
        self, hidden_states, encoder_hidden_states,
        embedded_timestep, frames, height, width, video_rotary_emb
    ):
        layer_idx = sum([len(stage_block) for stage_block in self.transformer_blocks[:len(self.num_layers) // 2]])
        for idx_, block in enumerate(self.transformer_blocks[len(self.num_layers) // 2]):
            layer_idx += 1
            attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
            hidden_states, encoder_hidden_states = self.block_forward(
                block=block,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                frames=frames,
                height=height,
                width=width,
                video_rotary_emb=video_rotary_emb,
                layer_idx=layer_idx
            )
        return hidden_states, encoder_hidden_states
        
    def _operate_on_dec(
        self, hidden_states, skip_connections, encoder_hidden_states,
        embedded_timestep, frames, height, width, video_rotary_emb
    ):
        layer_idx = sum([len(stage_block) for stage_block in self.transformer_blocks[:len(self.num_layers) // 2 + 1]])
        for idx, stage_block in enumerate(self.transformer_blocks[-(len(self.num_layers) // 2):]):
            if self.skip_connection:
                skip_hidden_states, skip_encoder_hidden_states = skip_connections.pop()
                hidden_states = torch.cat([hidden_states, skip_hidden_states], dim=-1)
                hidden_states = self.skip_norm_linear[idx](hidden_states)
                encoder_hidden_states = torch.cat([encoder_hidden_states, skip_encoder_hidden_states], dim=-1)
                encoder_hidden_states = self.skip_norm_linear_enc[idx](encoder_hidden_states)
            
            for idx_, block in enumerate(stage_block):
                layer_idx += 1
                attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
                hidden_states, encoder_hidden_states = self.block_forward(
                    block=block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    frames=frames,
                    height=height,
                    width=width,
                    video_rotary_emb=video_rotary_emb,
                    layer_idx=layer_idx
                )
                
        return hidden_states, encoder_hidden_states

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, pooled_projections):

        hidden_states = self.patch_embed(hidden_states.to(self.dtype))
        if pooled_projections.shape[1] != 1:
            raise AssertionError("Pooled projection should have shape (b, 1, 1, d)")
        pooled_projections = pooled_projections.squeeze(1)  # b 1 1 d -> b 1 d
        timesteps_emb = self.time_text_embed(timestep, pooled_projections)  # (N, D)
            
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d
        if encoder_hidden_states.shape[1] != 1:
            raise AssertionError("Encoder hidden states should have shape (b, 1, l, d)")
        encoder_hidden_states = encoder_hidden_states.squeeze(1)

        return hidden_states, encoder_hidden_states, timesteps_emb

    def _get_output_for_patched_inputs(
        self, hidden_states, embedded_timestep, frames, height, width
    ):
        hidden_states = self.norm_final(hidden_states)

        hidden_states = self.norm_out(hidden_states, temb=embedded_timestep)

        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states,
                                                                           tensor_parallel_output_grad=False)

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=hidden_states.shape[1]).contiguous()

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            -1, frames, height, width,
            self.patch_size_t, self.patch_size, self.patch_size, self.out_channels
        )

        hidden_states = torch.einsum("nthwopqc -> nctohpwq", hidden_states)
        output = hidden_states.reshape(
            -1,
            self.out_channels,
            frames * self.patch_size_t,
            height * self.patch_size,
            width * self.patch_size
        )
        return output
    
    def block_forward(self, block, hidden_states, attention_mask, encoder_hidden_states, embedded_timestep, frames, height, width, video_rotary_emb, layer_idx):
        if self.training and layer_idx <= self.recompute_num_layers and not selective_recom:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                embedded_timestep,
                frames,
                height,
                width,
                video_rotary_emb,
                **ckpt_kwargs
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                frames=frames,
                height=height,
                width=width,
                video_rotary_emb=video_rotary_emb,
                recom_ffn=layer_idx <= recom_ffn_layers
            )
        return hidden_states, encoder_hidden_states

class SparseMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        timestep_embed_dim: int, 
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = False,
        context_pre_only: bool = False,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1),
        double_ff: bool = False,
        sparse1d: bool = False,
        sparse_n: int = 2,
        sparse_group: bool = False,
        norm_cls: str = 'layer_norm',
    ):
        super().__init__()

        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.head_dim = head_dim

        self.tp_size = mpu.get_tensor_model_parallel_world_size()

        self.context_pre_only = context_pre_only
        self.double_ff = double_ff

        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = LayerNorm

        # adanorm-zero1: to introduce timestep and clip condition
        self.norm1 = OpenSoraNormZero(
            timestep_embed_dim, 
            dim, 
            norm_elementwise_affine,
            norm_eps, 
            bias=True, 
            norm_cls=norm_cls,
            context_pre_only=False, # we always need enc branch in norm1
        )

        # 1. MM Attention
        self.attn1 = MultiHeadSparseMMAttentionSBH(
            query_dim=dim,
            key_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            added_kv_proj_dim=dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            context_pre_only=context_pre_only,
            qk_norm='rms_norm',
            eps=norm_eps,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=False,
        )

        context_pre_only_for_norm2_ff_enc = True
        if self.double_ff:
            if self.context_pre_only is not None:
                if self.tp_size > 1 or not self.context_pre_only:
                    context_pre_only_for_norm2_ff_enc = False

        # adanorm-zero2: to introduce timestep and clip condition
        self.norm2 = OpenSoraNormZero(
            timestep_embed_dim,
            dim, 
            norm_elementwise_affine, 
            norm_eps, 
            bias=True, 
            norm_cls=norm_cls,
            context_pre_only=context_pre_only_for_norm2_ff_enc,
        )

        # 2. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.ff_enc = None
        if not context_pre_only_for_norm2_ff_enc:
            self.ff_enc = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

    def ffn(self, hidden_states, encoder_hidden_states, embedded_timestep):
        vis_seq_length, batch_size = hidden_states.shape[:2]
         # 4. norm & scale & shift
        hidden_states = maybe_clamp_tensor(hidden_states, training=self.training)
        encoder_hidden_states = maybe_clamp_tensor(encoder_hidden_states, training=self.training)
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, embedded_timestep
        )
        # import ipdb; ipdb.set_trace()
        weight_dtype = hidden_states.dtype
        if self.double_ff:
            # 5. FFN
            vis_ff_output = self.ff(norm_hidden_states)
            # 6. residual & gate
            with torch.autocast("cuda", enabled=False):
                hidden_states = hidden_states.float() + gate_ff.float() * vis_ff_output.float()
            hidden_states = hidden_states.to(weight_dtype)
            if self.ff_enc is not None:
                enc_ff_output = self.ff_enc(norm_encoder_hidden_states)
                with torch.autocast("cuda", enabled=False):
                    encoder_hidden_states = encoder_hidden_states.float() + enc_gate_ff.float() * enc_ff_output.float()
                encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
        else:
            # 5. FFN
            norm_hidden_states = torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=0)
            ff_output = self.ff(norm_hidden_states)
            # 6. residual & gate
            with torch.autocast("cuda", enabled=False):
                hidden_states = hidden_states.float() + gate_ff.float() * ff_output[:vis_seq_length].float()
                encoder_hidden_states = encoder_hidden_states.float() + enc_gate_ff.float() * ff_output[vis_seq_length:].float()
            hidden_states = hidden_states.to(weight_dtype)
            encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        embedded_timestep: Optional[torch.LongTensor] = None,
        frames: int = None, 
        height: int = None, 
        width: int = None,
        video_rotary_emb: Optional[torch.FloatTensor] = None,
        recom_ffn = False,
    ) -> torch.FloatTensor:
        # 1. norm & scale & shift
        hidden_states = maybe_clamp_tensor(hidden_states, training=self.training)
        encoder_hidden_states = maybe_clamp_tensor(encoder_hidden_states, training=self.training)
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, embedded_timestep
        )        

        # print('norm1')
        # print(f'norm_hidden_states: {norm_hidden_states.shape}, norm_encoder_hidden_states: {norm_encoder_hidden_states.shape}, gate_msa: {gate_msa.shape}, enc_gate_msa: {enc_gate_msa.shape}')
        # 2. MM Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            frames=frames,
            height=height,
            width=width,
            attention_mask=attention_mask,
            video_rotary_emb=video_rotary_emb,
        )
        # print('attn1')
        # print(f'attn_hidden_states: {attn_hidden_states.shape}, attn_encoder_hidden_states: {attn_encoder_hidden_states.shape}')
        # 3. residual & gate
        weight_dtype = hidden_states.dtype
        with torch.autocast("cuda", enabled=False):
            hidden_states = hidden_states.float() + gate_msa.float() * attn_hidden_states.float()
        hidden_states = hidden_states.to(weight_dtype)
        if self.context_pre_only is not None:
            if self.tp_size > 1 or not self.context_pre_only:
                with torch.autocast("cuda", enabled=False):
                    encoder_hidden_states = encoder_hidden_states.float() + enc_gate_msa.float() * attn_encoder_hidden_states.float()
                encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

        # import ipdb; ipdb.set_trace()
        if self.training and recom_ffn:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.ffn),
                hidden_states, encoder_hidden_states, embedded_timestep,
                **ckpt_kwargs
            )
        else:
            hidden_states, encoder_hidden_states = self.ffn(
                hidden_states, encoder_hidden_states, embedded_timestep
            )


        return hidden_states, encoder_hidden_states

