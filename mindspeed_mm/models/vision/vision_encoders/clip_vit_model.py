# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Union

import torch
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.utils.utils import get_device

from .vision_transformer_block import VisionTransformerBlock


class CLIPViT(MultiModalModule):
    """
    CLIP ViT vision model.
    Instantiate a CLIP Vit model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_size (int): Image patch size.
        image_size (int): Input image size.
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
    ) -> None:
        super().__init__(config=config)
        self.device = get_device(config.device)
        self.class_token_len = config.class_token_len
        self.visual_hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.img_h = config.image_size
        self.img_w = config.image_size

        if self.img_h % self.patch_size != 0:
            raise AssertionError("patch_size shoule be an exact divisor of img_height")
        if self.img_w % self.patch_size != 0:
            raise AssertionError("patch_size shoule be an exact divisor of img_width")
        self.num_patches_per_dim_h = self.img_h // self.patch_size
        self.num_patches_per_dim_w = self.img_w // self.patch_size
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = config.add_class_token
        self.class_token_len = config.class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.position_ids = torch.arange(self.seq_length).expand(1, -1).to(self.device)

        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

        if self.add_class_token:
            self.class_token = torch.nn.Parameter(
                torch.randn(1, self.class_token_len, self.visual_hidden_size)
            )

        self.ln_pre = TENorm(
            config=self.config,
            hidden_size=self.visual_hidden_size,
        )

        self.model_type = ModelType.encoder_or_decoder

        self.decoder = VisionTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """
        Sets pinput tensor to the model.

        Args:
            input_tensor (torch.Tensor):Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): Input data of shape [batch, img_h, img_w]
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use. If none, all ones.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        x = self.conv1(x)  # [batch, hidden_size, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]
            x = torch.cat(
                [class_token, x], dim=1
            )  # [batch, grid ** 2 + class_token_len, hidden_size]

        if x.shape[1] != self.seq_length:
            raise AssertionError(f"{x.shape[1]} != {self.seq_length}")
        x = x + self.position_embeddings(self.position_ids)
        x = self.ln_pre(x)
        # contiguous() call required as 'permute' can sparsify the tensor and this breaks pipelining
        x = x.permute(1, 0, 2).contiguous()  # [b, s, h] -> [s, b, h],
        if attention_mask is None:
            attention_mask = torch.ones(
                1, 1, self.seq_length, self.seq_length
            ).to(self.device)  # [1, 1, s, s]
            attention_mask = attention_mask < 0.5  # to bool

        x = self.decoder(x, attention_mask)
        x = x.permute(1, 0, 2).contiguous()  # [s, b, h] -> [b, s, h]
        x = x[:, 1:]

        return x
