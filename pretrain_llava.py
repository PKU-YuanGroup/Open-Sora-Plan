# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain LLaVA."""
from copy import deepcopy
import dataclasses

import torch
import torch.nn.functional as F
import mindspeed.megatron_adaptor

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.transformer import TransformerConfig
from megatron.training import get_args, print_rank_0, get_timers
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.models.vl_model import VLModel
from mindspeed_mm.training import pretrain
from mindspeed_mm.configs.config import MMConfig
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec, get_mlp_module_spec


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    vlm_config = deepcopy(args.mm.model)
    print_rank_0("building LLaVA model ...")
    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)
    vlm_config.text_decoder.language_tansformer_layer_spec = get_layer_spec(is_vit=False)
    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_encoder.vision_transformer_layer_spec = get_layer_spec(is_vit=True)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)
    vlm_config.image_encoder.vision_projector.vision_projection_layer_spec = get_mlp_module_spec(use_te=False).submodules
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process
    model = VLModel(vlm_config)
    model.freeze(vlm_config.text_decoder.freeze, vlm_config.image_encoder.vision_encoder.freeze, vlm_config.image_encoder.vision_projector.freeze)
    return model


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    images = data["pixel_values"].to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    input_ids = data["input_ids"].to(device=torch.cuda.current_device())
    labels = data["labels"].to(device=torch.cuda.current_device())
    attention_mask = data["attention_mask"].to(device=torch.cuda.current_device())

    return images, input_ids, labels, attention_mask


def loss_func(output_tensor):
    """Loss function."""
    averaged_loss = average_losses_across_data_parallel_group([output_tensor])
    loss = output_tensor.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    images, input_ids, labels, attention_mask = get_batch(data_iterator)
    timers("batch-generator").stop()
    position_ids = None
    output_tensor = model(
        images,
        input_ids,
        position_ids,
        attention_mask,
        labels
    )
    
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
    )
    return iter(train_dataloader), None, None


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False},
    )
