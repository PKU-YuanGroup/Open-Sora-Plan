# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn

from .projectors.multimodal_projector import MultimodalProjector
from .vision_encoders.clip_vit_model import CLIPViT

VISION_MODEL_MAPPINGS = {
    "clip": CLIPViT,
    "projector": MultimodalProjector
}


class VisionModel(nn.Module):
    """
    Instantiate a vision encoder model from config.

    Args:
        config (dict): the general config for Vision Model
        {
            "vision_encoder": {...},  # Config for the image encoder.
            "vision_projector": {...},  # Config for the image projector.
            "drop_vision_class_token": (bool),  # Drop vision class token(s) before input to the text decoder.
        }
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = VISION_MODEL_MAPPINGS[config.vision_encoder.model_id](config.vision_encoder)
        self.projector = VISION_MODEL_MAPPINGS[config.vision_projector.model_id](config.vision_projector)
        self._drop_vision_class_token = config.drop_vision_class_token

    def get_model(self):
        return self.encoder, self.projector

    def freeze(
        self,
        freeze_encoder: bool = False,
        freeze_projector: bool = False
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_encoder (bool): Freeze the image encoder module.
            freeze_projection (bool): Freeze the image projector module.
        """

        modules = []
        if freeze_encoder and self.encoder is not None:
            modules.append(self.encoder)
        if freeze_projector and self.projector is not None:
            modules.append(self.projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.encoder(images)  # [b, img_seq_len, h_vision]
        if self._drop_vision_class_token:
            image_embeddings = image_embeddings[:, self.encoder.class_token_len :, :]
        image_embeddings = image_embeddings.permute(1, 0, 2).contiguous()  # [img_seq_len, b, h_vision]
        image_embeddings = self.projector(image_embeddings)  # [img_seq_len, b, h_vision]

        return image_embeddings
    