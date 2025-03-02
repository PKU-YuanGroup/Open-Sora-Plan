# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
import importlib
from functools import lru_cache
from einops import rearrange

import torch
import torch.distributed


@lru_cache
def is_npu_available():
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu
    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


def get_device(device="npu"):
    """
    only support npu and cpu device, default npu.
    device format: cpu, npu, or npu:0
    """
    if isinstance(device, torch.device):
        return device
    device = device.lower().strip()
    if device == "cpu":
        return torch.device(device)

    device_infos = device.split(":")
    device_name = device_infos[0]
    if device_name == "npu":
        if is_npu_available():
            if len(device_infos) == 1:
                return torch.device(device_name)
            if len(device_infos) == 2:
                device_id = int(device_infos[1])
                num_devices = torch.npu.device_count()
                if device_id < num_devices:
                    return torch.device(f"{device_name}:{device_id}")
                else:
                    raise ValueError(f"device_id: {device_id} must less than device nums: {num_devices}")
        else:
            raise RuntimeError("NPU environment is not available")
    raise ValueError("only support npu and cpu device. device format: cpu, npu, or npu:0")


def get_dtype(dtype):
    """return torch type according to the string"""
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_mapping = {
        "int32": torch.int32,
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_mapping:
        raise ValueError("Unsupported data type")
    dtype = dtype_mapping[dtype]
    return dtype


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = func(self, x, *args, **kwargs)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x
    return wrapper


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) or isinstance(t, list) else ((t,) * length)


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_SIZE = None


def is_context_parallel_initialized():
    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    else:
        return True


def set_context_parallel_group(size, group):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE
    _CONTEXT_PARALLEL_GROUP = group
    _CONTEXT_PARALLEL_SIZE = size


def initialize_context_parallel(context_parallel_size):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    if _CONTEXT_PARALLEL_GROUP is not None:
        raise AssertionError("Context parallel group is already initialized")
    _CONTEXT_PARALLEL_SIZE = context_parallel_size

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            break


def get_context_parallel_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise AssertionError("Context parallel group is not initialized")

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size():
    if _CONTEXT_PARALLEL_SIZE is None:
        raise AssertionError("Context parallel size is not initialized")

    return _CONTEXT_PARALLEL_SIZE


def get_context_parallel_rank():
    if _CONTEXT_PARALLEL_SIZE is None:
        raise AssertionError("Context parallel size is not initialized")

    rank = torch.distributed.get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank


def get_context_parallel_group_rank():
    if _CONTEXT_PARALLEL_SIZE is None:
        raise AssertionError("Context parallel size is not initialized")

    rank = torch.distributed.get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank
