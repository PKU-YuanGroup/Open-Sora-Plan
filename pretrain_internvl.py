# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain InternVL."""

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import mindspeed.megatron_adaptor
from mindspeed.utils import get_batch_on_this_cp_rank

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.utils.utils import get_dtype
from mindspeed_mm.models.internvl_model import InternVLModel
from mindspeed_mm.utils.transformer_model_config import get_model_config


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building InternVL model ...")
    model_config = args.mm.model
    model_config.image_encoder.vision_encoder = get_model_config(model_config.image_encoder.vision_encoder)
    model_config.text_decoder = get_model_config(model_config.text_decoder)

    model = InternVLModel(model_config)

    return model


def get_batch_on_this_tp_rank(data_iterator):
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                        group=mpu.get_tensor_model_parallel_group()
                                        )
            
    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            batch = None

        input_ids = batch['input_ids'].to(torch.cuda.current_device())
        labels = batch['labels'].to(torch.cuda.current_device())
        attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
        image = batch['pixel_values'].to(torch.cuda.current_device())
        image_flags = batch['image_flags'].to(torch.cuda.current_device())
        _broadcast(input_ids)
        _broadcast(labels)
        _broadcast(attention_mask)
        _broadcast(image)
        _broadcast(image_flags)

    else:
        raise NotImplementedError
    
    batch = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'image': image,
        'image_flags': image_flags
    }

    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        batch = get_batch_on_this_cp_rank(batch)
        return batch['input_ids'], batch['labels'], batch['attention_mask'], batch['image'], batch['image_flags']
    else:
        return None, None, None, None, None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor['loss'].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    input_ids, labels, attention_mask, image, image_flags = get_batch(data_iterator)
    if image is not None:
        image = image.to(args.params_dtype)
    output = model(
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_flags=image_flags
    )
    return output, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        data_config.dataloader_param
    )
    train_dataloader, val_dataloader, test_dataloader = build_iterations(train_dataloader)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", 
                       "vision_pretraining": False}
    )
