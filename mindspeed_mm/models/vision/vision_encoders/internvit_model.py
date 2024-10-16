# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.utils.utils import get_device


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, config=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class InternVitTransformerLayer(TransformerLayer):
    """
        A single transformer layer for Intern ViT.

        TransformerLayer takes input with size [s, b, h] and returns an
        output of the same size.
    """
    def __init__(
        self, 
        config: TransformerConfig, 
        submodules: TransformerLayerSubmodules, 
        layer_number: int = 1, 
        hidden_dropout: float = None
    ):
        super().__init__(config=config, 
                         submodules=submodules, 
                         layer_number=layer_number, 
                         hidden_dropout=hidden_dropout)
        
        if config.normalization == 'layernorm':
            self.input_layernorm = torch.nn.LayerNorm(config.hidden_size, config.layernorm_epsilon)
            self.pre_mlp_layernorm = torch.nn.LayerNorm(config.hidden_size, config.layernorm_epsilon)

        # InternViT新增可学习参数
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(config.hidden_size))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(config.hidden_size))

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        context=None, 
        context_mask=None, 
        rotary_pos_emb=None, 
        inference_params=None, 
        packed_seq_params=None
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        attention_output = (attention_output_with_bias[0] + attention_output_with_bias[1]) * self.ls1
        attention_output_with_bias = (attention_output, None)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        mlp_output = (mlp_output_with_bias[0] + mlp_output_with_bias[1]) * self.ls2
        mlp_output_with_bias = (mlp_output, None)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context


class InternVitSelfAttention(SelfAttention):
    """
        Self-attention layer class for InternVit

        Self-attention layer takes input with size [s, b, h]
        and returns output of the same size.
    """

    def __init__(
        self, 
        config: TransformerConfig, 
        submodules: SelfAttentionSubmodules, 
        layer_number: int, 
        attn_mask_type=AttnMaskType.padding
    ):
        super().__init__(
            config=config, 
            submodules=submodules, 
            layer_number=layer_number, 
            attn_mask_type=attn_mask_type
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.config.hidden_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None
        
    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        N, B, C = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(N, B, 3, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        mixed_qkv = mixed_qkv.permute(2, 0, 3, 1, 4)
        
        query, key, value = mixed_qkv.unbind(0)  # [sq, np, b, hn]
        
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        if self.q_layernorm is not None:
            N_, H_, B_, D_ = query.shape
            query = self.q_layernorm(query.permute(2, 0, 1, 3).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(0, 1)

        if self.k_layernorm is not None:
            N_, H_, B_, D_ = key.shape
            key = self.k_layernorm(key.permute(2, 0, 1, 3).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(0, 1)

        value = value.permute(0, 2, 1, 3)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config, image_size=448, patch_size=14):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternViT(MultiModalModule):

    def __init__(self, config: TransformerConfig,
                 transformer_layer_spec: ModuleSpec,
                 image_size=448,
                 patch_size=14):
        super().__init__(config)
        self.config = config
        self.image_size = image_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.embeddings = InternVisionEmbeddings(config, self.image_size, self.patch_size)
        self.encoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False
        )

    def set_input_tensor(self, input_tensor):
        return super().set_input_tensor(input_tensor)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        print('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings
    
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def extract_feature(self, hidden_states):
        vit_embeds = hidden_states[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds
    
    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ):

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        
        seq_length = hidden_states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(
                1, 1, seq_length, seq_length
            ).to(hidden_states.device)
            attention_mask = attention_mask < 0.5

        hidden_states = hidden_states.transpose(0, 1)
        encoder_outputs = self.encoder(hidden_states, attention_mask)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        if self.select_layer == -1:
            vit_embeds = encoder_outputs
        else:
            vit_embeds = encoder_outputs[self.select_layer]
        
        vit_embeds = self.extract_feature(vit_embeds)

        return vit_embeds
