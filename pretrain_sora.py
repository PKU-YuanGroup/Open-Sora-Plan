# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain SoRA."""

import torch

import mindspeed.megatron_adaptor

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    unwrap_model,
)

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import VIDEO, PROMPT_IDS, PROMPT_MASK, VIDEO_MASK
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.sora_model import SoRAModel


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building SoRA model ...")
    model = SoRAModel(args.mm.model)
    return model


def get_batch_on_this_tp_rank(data_iterator):
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(torch.cuda.current_device())
    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch
    else:
        return None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor.mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    batch = get_batch(data_iterator)
    video = batch.pop(VIDEO, None)
    prompt_ids = batch.pop(PROMPT_IDS, None)
    video_mask = batch.pop(VIDEO_MASK, None)
    prompt_mask = batch.pop(PROMPT_MASK, None)
    output_tensor_list = model(video, prompt_ids, video_mask, prompt_mask=prompt_mask, **batch)
    loss_dict = unwrap_model(model).compute_loss(*output_tensor_list)
    return loss_dict, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
    )
    data_iterator, _, _ = build_iterations(train_dl=train_dataloader)
    return data_iterator, None, None


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
