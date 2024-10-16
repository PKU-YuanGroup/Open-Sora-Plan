#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd
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

import gc

import torch
import torch.distributed as dis
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


def compute_vae_encode(batch, accelerator, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    return {"model_input": accelerator.gather(model_input)}


class TorchPatcher:

    @staticmethod
    def new_get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the asccociated process group
        This patch method makes the torch npu available for distribution
        """
        if dist.get_backend(self._process_group) == dist.Backend.NCCL:
            return torch.device(torch.cuda.current_device())
        try:
            import torch_npu

            return torch.device(torch_npu.npu.current_device())
        except Exception as e:
            return torch.device("cpu")

    @classmethod
    def apply_patch(cls):
        # Apply the patch for npu distribution
        ShardedTensor._get_preferred_device = cls.new_get_preferred_device


def config_gc():
    # set gc threshold
    gc.set_threshold(700, 50, 1000)
