# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pretrain Whisper."""
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import mindspeed.megatron_adaptor
import torch
from datasets import Audio, load_dataset
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    unwrap_model,
)
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    PROMPT_IDS,
    PROMPT_MASK,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.data.dataloader.sampler import StatefulDistributedSampler
from mindspeed_mm.models.whisper.whisper_model import WhisperForConditionalGeneration_mm
from mindspeed_mm.training import pretrain


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building whisper model ...")
    model = WhisperForConditionalGeneration_mm(args.mm.model)
    return model


def get_batch_on_this_tp_rank(data_iterator):
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None
    labels = batch["labels"].to(torch.cuda.current_device())
    input_features = batch["input_features"].to(torch.cuda.current_device())
    batch = {"input_features": input_features, "labels": labels}
    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch["input_features"], batch["labels"]
    else:
        return None, None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor.mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    input_features, labels = get_batch(data_iterator)
    output = model(input_features, labels)
    loss_dict = unwrap_model(model).compute_loss(output, labels)
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
