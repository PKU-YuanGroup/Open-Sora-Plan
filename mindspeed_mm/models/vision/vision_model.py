# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn

from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec, get_mlp_module_spec
from .projectors.multimodal_projector import MultimodalProjector
from .vision_encoders.clip_vit_model import CLIPViT


VISION_MODEL_MAPPINGS = {
    "clip": CLIPViT,
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
    def __init__(self, config):
        super().__init__()
        vision_transformer_layer_spec = get_layer_spec(is_vit=True)
        self.encoder = VISION_MODEL_MAPPINGS[config.vision_encoder.model_id](
            config.vision_encoder,
            vision_transformer_layer_spec,
            add_class_token=config.add_class_token,
            class_token_len=config.class_token_len,
            image_size=config.image_size,
            patch_size=config.patch_size,
        )
        self.encoder.requires_grads_(False)
        vision_projection_layer_spec = get_mlp_module_spec(use_te=False).submodules
        self.projector = VISION_MODEL_MAPPINGS[config.vision_projector.model_id](
            config.vision_projector,
            vision_projection_layer_spec,
            config.vision_projector.model_id,
            config.vision_encoder.hidden_size,
        )
        self.projector.requires_grads_(True)

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
        image_embeddings = self.projector(image_embeddings)

        return image_embeddings
    