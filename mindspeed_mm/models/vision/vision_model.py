# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn

from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec, get_mlp_module_spec
from .projectors.multimodal_projector import MultimodalProjector
from .vision_encoders.clip_vit_model import CLIPViT
from .vision_encoders.internvit_model import InternViT


VISION_MODEL_MAPPINGS = {
    "clip": CLIPViT,
    "InternViT": InternViT,
    "mlp": MultimodalProjector
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
    def __init__(self, config, encoder_transformer_layer_spec=None, projector_layer_spec=None):
        super().__init__()
        self.add_projector = config.vision_projector is not None
        self.encoder = VISION_MODEL_MAPPINGS[config.vision_encoder.model_id](
            config.vision_encoder,
            encoder_transformer_layer_spec
        )
        if self.add_projector:
            self.projector = VISION_MODEL_MAPPINGS[config.vision_projector.model_id](
                config.vision_projector,
                projector_layer_spec
            )

    def set_input_tensor(self, input_tensor):
        self.encoder.set_input_tensor(input_tensor)

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
        image_embeddings = self.encoder(images)
        if self.add_projector:
            image_embeddings = self.projector(image_embeddings)

        return image_embeddings
    