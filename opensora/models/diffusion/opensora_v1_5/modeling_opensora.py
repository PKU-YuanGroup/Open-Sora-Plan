import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple, List
from torch.nn import functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.opensora_v1_5.modules import CombinedTimestepTextProjEmbeddings, BasicTransformerBlock, AdaNorm
from opensora.utils.utils import to_2tuple
from opensora.models.diffusion.common import PatchEmbed2D
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info



def prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n, head_num):
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

    attention_mask_sparse_1d = torch.cat([attention_mask_sparse_1d, encoder_attention_mask_sparse], dim=-1)
    attention_mask_sparse_1d_group = torch.cat([attention_mask_sparse_1d_group, encoder_attention_mask_sparse], dim=-1)

    if npu_config is not None:
        attention_mask_sparse_1d = npu_config.get_attention_mask(
            attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1]
            )
        attention_mask_sparse_1d_group = npu_config.get_attention_mask(
            attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1]
            )
    else:
        attention_mask_sparse_1d = attention_mask_sparse_1d.repeat_interleave(head_num, dim=1)
        attention_mask_sparse_1d_group = attention_mask_sparse_1d_group.repeat_interleave(head_num, dim=1)

    return {
                False: attention_mask_sparse_1d,
                True: attention_mask_sparse_1d_group
            }



                

class OpenSoraT2V_v1_5(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: List[int] = [2, 4, 8, 4, 2], 
        sparse_n: List[int] = [1, 4, 16, 4, 1], 
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
        norm_cls: str = 'rms_norm', 
        skip_connection: bool = False
    ):
        super().__init__()
        # Set some common variables used across the board.
        self.out_channels = in_channels if out_channels is None else out_channels
        self.config.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False
        if norm_cls == 'rms_norm':
            self.norm_cls = RMSNorm
        elif norm_cls == 'layer_norm':
            self.norm_cls = nn.LayerNorm

        assert len(self.config.num_layers) == len(self.config.sparse_n)
        assert len(self.config.num_layers) % 2 == 1
        assert all([i % 2 == 0 for i in self.config.num_layers])

        if not self.config.sparse1d:
            self.config.sparse_n = self.sparse_n = [1] * len(self.config.sparse_n)

        self._init_patched_inputs()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _init_patched_inputs(self):

        # 0. some param
        self.config.sample_size = (self.config.sample_size_h, self.config.sample_size_w)
        interpolation_scale_thw = (
            self.config.interpolation_scale_t, 
            self.config.interpolation_scale_h, 
            self.config.interpolation_scale_w
            )
        
        # 1. patch embedding
        self.patch_embed = PatchEmbed2D(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )
        
        # 2. time embedding and pooled text embedding
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            timestep_embed_dim=self.config.timestep_embed_dim, 
            embedding_dim=self.config.timestep_embed_dim, 
            pooled_projection_dim=self.config.pooled_projection_dim
        )

        # 3. anthor text embedding
        self.caption_projection = nn.Linear(self.config.caption_channels, self.config.hidden_size)

        # forward transformer blocks
        self.transformer_blocks = []
        self.skip_norm_linear = []
        self.skip_norm_linear_enc = []
        for idx, (num_layer, sparse_n) in enumerate(zip(self.config.num_layers, self.config.sparse_n)):
            is_last_stage = idx == len(self.config.num_layers) - 1
            if self.config.skip_connection and idx > len(self.config.num_layers) // 2:
                self.skip_norm_linear.append(
                    nn.Sequential(
                        self.norm_cls(
                            self.config.hidden_size*2, 
                            elementwise_affine=self.config.norm_elementwise_affine, 
                            eps=self.config.norm_eps
                            ), 
                        nn.Linear(self.config.hidden_size*2, self.config.hidden_size), 
                    )
                )
                self.skip_norm_linear_enc.append(
                    nn.Sequential(
                        self.norm_cls(
                            self.config.hidden_size*2, 
                            elementwise_affine=self.config.norm_elementwise_affine, 
                            eps=self.config.norm_eps
                            ), 
                        nn.Linear(self.config.hidden_size*2, self.config.hidden_size), 
                    )
                )
            stage_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        self.config.hidden_size,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        self.config.timestep_embed_dim, 
                        dropout=self.config.dropout,
                        activation_fn=self.config.activation_fn,
                        attention_bias=self.config.attention_bias,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                        interpolation_scale_thw=interpolation_scale_thw, 
                        sparse1d=self.config.sparse1d if sparse_n > 1 else False, 
                        sparse_n=sparse_n, 
                        sparse_group=i % 2 == 1 if sparse_n > 1 else False, 
                        context_pre_only=is_last_stage and (i == num_layer - 1), 
                        # context_pre_only=True, 
                        norm_cls=self.config.norm_cls, 
                    )
                    for i in range(num_layer)
                ]
            )
            self.transformer_blocks.append(stage_blocks)
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        if self.config.skip_connection:
            self.skip_norm_linear = nn.ModuleList(self.skip_norm_linear)
            self.skip_norm_linear_enc = nn.ModuleList(self.skip_norm_linear_enc)

        # norm out and unpatchfy
        self.norm_final = self.norm_cls(
            self.config.hidden_size, eps=self.config.norm_eps, 
            elementwise_affine=self.config.norm_elementwise_affine
            )
        self.norm_out = AdaNorm(
            embedding_dim=self.config.timestep_embed_dim,
            output_dim=self.config.hidden_size * 2,  # shift and scale
            norm_elementwise_affine=self.config.norm_elementwise_affine,
            norm_eps=self.config.norm_eps,
            chunk_dim=1,
            norm_cls=self.config.norm_cls, 
        )
        self.proj_out = nn.Linear(
            self.config.hidden_size, self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
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
            # b, frame, h, w -> a video with images
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
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0


        # 1. Input
        frame = ((frame - 1) // self.config.patch_size_t + 1) if frame % 2 == 1 else frame // self.config.patch_size_t  # patchfy
        height, width = hidden_states.shape[-2] // self.config.patch_size, hidden_states.shape[-1] // self.config.patch_size


        hidden_states, encoder_hidden_states, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, pooled_projections
        )
        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()

        self.sparse_mask = {}
        if npu_config is None:
            if get_sequence_parallel_state():
                head_num = self.config.num_attention_heads // nccl_info.world_size
            else:
                head_num = self.config.num_attention_heads
        else:
            head_num = None
        for sparse_n in list(set(self.config.sparse_n)):
            self.sparse_mask[sparse_n] = prepare_sparse_mask(
                attention_mask, encoder_attention_mask, sparse_n, head_num
                )

        # 2. Blocks
        hidden_states, encoder_hidden_states, skip_connections = self._operate_on_enc(
            hidden_states, encoder_hidden_states, 
            embedded_timestep, frame, height, width
            )
        
        hidden_states, encoder_hidden_states = self._operate_on_mid(
            hidden_states, encoder_hidden_states, 
            embedded_timestep, frame, height, width
            )
        
        hidden_states, encoder_hidden_states = self._operate_on_dec(
            hidden_states, skip_connections, encoder_hidden_states, 
            embedded_timestep, frame, height, width
            )

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states, embedded_timestep, frame, height=height, width=width,
        )  # b c t h w

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _operate_on_enc(
            self, hidden_states, encoder_hidden_states, 
            embedded_timestep, frame, height, width
        ):
        
        skip_connections = []
        for idx, stage_block in enumerate(self.transformer_blocks[:len(self.config.num_layers)//2]):
            for idx_, block in enumerate(stage_block):
                # print(f'------------enc stage_block_{idx}, block_{idx_}------------------')
                # print(f'enc stage_block_{idx}, block_{idx_} ', 
                #       f'sparse1d {block.sparse1d}, sparse_n {block.sparse_n}, sparse_group {block.sparse_group} '
                #       f'sparse_mask {block.sparse_n}, sparse_group {block.sparse_group}')
                attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        embedded_timestep,
                        frame, 
                        height, 
                        width, 
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        embedded_timestep=embedded_timestep,
                        frame=frame, 
                        height=height, 
                        width=width, 
                    )
                # print(f'enc hidden_states, block_{idx_} ', 
                #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
                # print(f'enc encoder_hidden_states, block_{idx_} ', 
                #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
            if self.config.skip_connection:
                skip_connections.append([hidden_states, encoder_hidden_states])
        # import sys;sys.exit()
        return hidden_states, encoder_hidden_states, skip_connections

    def _operate_on_mid(
            self, hidden_states, encoder_hidden_states, 
            embedded_timestep, frame, height, width
        ):
        
        for idx_, block in enumerate(self.transformer_blocks[len(self.config.num_layers)//2]):
            # print(f'------------mid block_{idx_}------------------')
            # print(f'mid, block_{idx_} ', 
            #         f'sparse1d {block.sparse1d}, sparse_n {block.sparse_n}, sparse_group {block.sparse_group} '
            #         f'sparse_mask {block.sparse_n}, sparse_group {block.sparse_group}')
            attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
            if self.training and self.gradient_checkpointing:

                
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    embedded_timestep,
                    frame, 
                    height, 
                    width, 
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    frame=frame, 
                    height=height, 
                    width=width, 
                )
            # print(f'mid hidden_states, block_{idx_} ', 
            #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
            # print(f'mid encoder_hidden_states, block_{idx_} ', 
            #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        
        return hidden_states, encoder_hidden_states


    def _operate_on_dec(
            self, hidden_states, skip_connections, encoder_hidden_states, 
            embedded_timestep, frame, height, width
        ):
        
        for idx, stage_block in enumerate(self.transformer_blocks[-(len(self.config.num_layers)//2):]):
            if self.config.skip_connection:
                skip_hidden_states, skip_encoder_hidden_states = skip_connections.pop()
                # print("hidden_state:",  "mean", hidden_states.mean(), "std", hidden_states.std())
                # print("skip_hidden_states:",  "mean", skip_hidden_states.mean(), "std", skip_hidden_states.std())
                hidden_states = torch.cat([hidden_states, skip_hidden_states], dim=-1)
                hidden_states = self.skip_norm_linear[idx](hidden_states)
                # print("encoder_hidden_states:",  "mean", encoder_hidden_states.mean(), "std", encoder_hidden_states.std())
                # print("skip_encoder_hidden_states:",  "mean", skip_encoder_hidden_states.mean(), "std", skip_encoder_hidden_states.std())
                encoder_hidden_states = torch.cat([encoder_hidden_states, skip_encoder_hidden_states], dim=-1)
                encoder_hidden_states = self.skip_norm_linear_enc[idx](encoder_hidden_states)
            for idx_, block in enumerate(stage_block):
                # print(f'------------dec stage_block_{idx}, block_{idx_}------------------')
                # print(f'dec stage_block_{idx}, block_{idx_} ', 
                #       f'sparse1d {block.sparse1d}, sparse_n {block.sparse_n}, sparse_group {block.sparse_group} '
                #       f'sparse_mask {block.sparse_n}, sparse_group {block.sparse_group}')
                attention_mask = self.sparse_mask[block.sparse_n][block.sparse_group]
                if self.training and self.gradient_checkpointing:

                    
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        embedded_timestep,
                        frame, 
                        height, 
                        width, 
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        embedded_timestep=embedded_timestep,
                        frame=frame, 
                        height=height, 
                        width=width, 
                    )
                # print(f'dec hidden_states, block_{idx_} ', 
                #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
                # print(f'dec encoder_hidden_states, block_{idx_} ', 
                #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        return hidden_states, encoder_hidden_states


    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, pooled_projections):
        
        # print(f'enc hidden_states, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'enc encoder_hidden_states, ', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        hidden_states = self.patch_embed(hidden_states.to(self.dtype))
        assert pooled_projections.shape[1] == 1
        pooled_projections = pooled_projections.squeeze(1)  # b 1 1 d -> b 1 d
        timesteps_emb = self.time_text_embed(timestep, pooled_projections)  # (N, D)
            
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d
        assert encoder_hidden_states.shape[1] == 1
        encoder_hidden_states = encoder_hidden_states.squeeze(1)
        # print('_operate_on_patched_inputs')
        # print(f'enc timesteps_emb, ', 
        #         f'max {timesteps_emb.max()}, min {timesteps_emb.min()}, mean {timesteps_emb.mean()}, std {timesteps_emb.std()}')
        # print(f'enc hidden_states, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # print(f'enc encoder_hidden_states, ', 
        #         f'max {encoder_hidden_states.max()}, min {encoder_hidden_states.min()}, mean {encoder_hidden_states.mean()}, std {encoder_hidden_states.std()}')
        # print('-----------------------')
        return hidden_states, encoder_hidden_states, timesteps_emb
    
    def _get_output_for_patched_inputs(
        self, hidden_states, embedded_timestep, num_frames, height, width
    ):  
        hidden_states = self.norm_final(hidden_states)
        # print(f'norm_final, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # Modulation
        hidden_states = self.norm_out(hidden_states, temb=embedded_timestep)
        # print(f'norm_out, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        # unpatchify
        hidden_states = self.proj_out(hidden_states)
        # print(f'proj_out, ', 
        #         f'max {hidden_states.max()}, min {hidden_states.min()}, mean {hidden_states.mean()}, std {hidden_states.std()}')
        hidden_states = hidden_states.reshape(
            -1, num_frames, height, width, 
            self.config.patch_size_t, self.config.patch_size, self.config.patch_size, self.out_channels
        )
        hidden_states = torch.einsum("nthwopqc->nctohpwq", hidden_states)
        output = hidden_states.reshape(
            -1, 
            self.out_channels, 
            num_frames * self.config.patch_size_t, 
            height * self.config.patch_size, 
            width * self.config.patch_size
        )
        # import sys;sys.exit()
        return output


def OpenSoraT2V_v1_5_2B_122(**kwargs):
    if kwargs.get('sparse_n', None) is not None:
        kwargs.pop('sparse_n')
    return OpenSoraT2V_v1_5(  # 22 layers
        num_layers=[2, 2, 4, 6, 4, 2, 2], sparse_n=[1, 2, 4, 8, 4, 2, 1], 
        attention_head_dim=96, num_attention_heads=24, 
        timestep_embed_dim=512, patch_size_t=1, patch_size=2, 
        caption_channels=2048, pooled_projection_dim=1280, **kwargs
    )

def OpenSoraT2V_v1_5_3B_122(**kwargs):
    if kwargs.get('sparse_n', None) is not None:
        kwargs.pop('sparse_n')
    return OpenSoraT2V_v1_5(  # 28 layers
        num_layers=[2, 4, 4, 8, 4, 4, 2], sparse_n=[1, 2, 4, 8, 4, 2, 1], 
        attention_head_dim=96, num_attention_heads=24, 
        timestep_embed_dim=768, patch_size_t=1, patch_size=2, 
        caption_channels=2048, pooled_projection_dim=1280, **kwargs
    )

def OpenSoraT2V_v1_5_6B_122(**kwargs):
    if kwargs.get('sparse_n', None) is not None:
        kwargs.pop('sparse_n')
    return OpenSoraT2V_v1_5(  # 32 layers
        num_layers=[2, 4, 6, 8, 6, 4, 2], sparse_n=[1, 2, 4, 8, 4, 2, 1], 
        attention_head_dim=96, num_attention_heads=32, 
        timestep_embed_dim=1024, patch_size_t=1, patch_size=2, 
        caption_channels=2048, pooled_projection_dim=1280, **kwargs
    )

def OpenSoraT2V_v1_5_9B_122(**kwargs):
    if kwargs.get('sparse_n', None) is not None:
        kwargs.pop('sparse_n')
    return OpenSoraT2V_v1_5(  # 32 layers
        num_layers=[2, 4, 6, 8, 6, 4, 2], sparse_n=[1, 2, 4, 8, 4, 2, 1], 
        attention_head_dim=96, num_attention_heads=40, 
        timestep_embed_dim=1280, patch_size_t=1, patch_size=2, 
        caption_channels=2048, pooled_projection_dim=1280, **kwargs
    )

def OpenSoraT2V_v1_5_13B_122(**kwargs):
    if kwargs.get('sparse_n', None) is not None:
        kwargs.pop('sparse_n')
    return OpenSoraT2V_v1_5(  # 40 layers
        num_layers=[2, 6, 8, 8, 8, 6, 2], sparse_n=[1, 2, 4, 8, 4, 2, 1], 
        attention_head_dim=128, num_attention_heads=32, 
        timestep_embed_dim=1536, patch_size_t=1, patch_size=2, 
        caption_channels=2048, pooled_projection_dim=1280, **kwargs
    )

def OpenSoraT2V_v1_5_32B_122(**kwargs):
    if kwargs.get('sparse_n', None) is not None:
        kwargs.pop('sparse_n')
    return OpenSoraT2V_v1_5(  # 48 layers
        num_layers=[4, 8, 8, 8, 8, 8, 4], sparse_n=[1, 2, 4, 8, 4, 2, 1], 
        attention_head_dim=144, num_attention_heads=40, 
        timestep_embed_dim=2048, patch_size_t=1, patch_size=2, 
        caption_channels=2048, pooled_projection_dim=1280, **kwargs
    )

OpenSora_v1_5_models = {
    "OpenSoraT2V_v1_5-2B/122": OpenSoraT2V_v1_5_2B_122, 
    "OpenSoraT2V_v1_5-3B/122": OpenSoraT2V_v1_5_3B_122, 
    "OpenSoraT2V_v1_5-6B/122": OpenSoraT2V_v1_5_6B_122, 
    "OpenSoraT2V_v1_5-9B/122": OpenSoraT2V_v1_5_9B_122, 
    "OpenSoraT2V_v1_5-13B/122": OpenSoraT2V_v1_5_13B_122, 
    "OpenSoraT2V_v1_5-32B/122": OpenSoraT2V_v1_5_32B_122, 
}

OpenSora_v1_5_models_class = {
    "OpenSoraT2V_v1_5-2B/122": OpenSoraT2V_v1_5,
    "OpenSoraT2V_v1_5-3B/122": OpenSoraT2V_v1_5,
    "OpenSoraT2V_v1_5-6B/122": OpenSoraT2V_v1_5,
    "OpenSoraT2V_v1_5-9B/122": OpenSoraT2V_v1_5,
    "OpenSoraT2V_v1_5-13B/122": OpenSoraT2V_v1_5,
    "OpenSoraT2V_v1_5-32B/122": OpenSoraT2V_v1_5,
}

if __name__ == '__main__':
    '''
    python opensora/models/diffusion/opensora_v1_5/modeling_opensora.py
    '''
    from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
    from opensora.models.causalvideovae import ae_norm, ae_denorm
    from opensora.models import CausalVAEModelWrapper
    args = type('args', (), 
    {
        'ae': 'WFVAEModel_D32_8x8x8', 
        'model_max_length': 300, 
        'max_height': 640,
        'max_width': 640,
        'num_frames': 105,
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
        "sparse1d": False, 
        "rank": 64, 
    }
    )
    b = 1
    c = 32
    cond_c = 2048
    cond_c1 = 1280
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    model = OpenSoraT2V_v1_5_8B_122(
        in_channels=c, 
        out_channels=c, 
        sample_size_h=latent_size, 
        sample_size_w=latent_size, 
        sample_size_t=num_frames, 
        norm_cls='rms_norm', 
        interpolation_scale_t=args.interpolation_scale_t, 
        interpolation_scale_h=args.interpolation_scale_h, 
        interpolation_scale_w=args.interpolation_scale_w, 
        sparse1d=args.sparse1d, 
        )
    print(model)
    total_cnt = len(list(model.named_parameters()))
    print('total_cnt', total_cnt)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    # import sys;sys.exit()
    try:
        # path = "/storage/ongoing/new/7.19anyres/Open-Sora-Plan/bs32x8x1_anyx93x640x640_fps16_lr1e-5_snr5_ema9999_sparse1d4_dit_l_mt5xxl_vpred_zerosnr/checkpoint-43000/model_ema/diffusion_pytorch_model.safetensors"
        # ckpt = torch.load(path, map_location="cpu")
        # msg = model.load_state_dict(ckpt, strict=True)
        print(msg)
    except Exception as e:
        print(e)
    model = model.to(device)
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1, args.model_max_length)).to(device)  # B 1 L
    timestep = torch.randint(0, 1000, (b,), device=device)
    pooled_projections = torch.randn(b, 1, cond_c1).to(device)
    model_kwargs = dict(hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask, pooled_projections=pooled_projections, 
                        encoder_attention_mask=cond_mask, timestep=timestep)
    with torch.no_grad():
        output = model(**model_kwargs)
    print(output[0].shape)
    # model.save_pretrained('./test_v1_5')

