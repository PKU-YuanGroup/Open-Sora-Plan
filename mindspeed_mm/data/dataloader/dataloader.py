# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from mindspeed_mm.data.data_utils.utils import get_seed_worker, collate_fn_default
from mindspeed_mm.data.dataloader.sampler import (
    LengthGroupedSampler,
    StatefulDistributedSampler,
)
from mindspeed_mm.data.dataloader.collator import DATA_COLLATOR

def prepare_base_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    collate_param=None,
    **kwargs,
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader`.

    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    collate_fn = None
    if collate_param is None:
        raise ValueError("collate_param must be provided.")
    data_collate_type = collate_param.pop("model_name")
    if data_collate_type is None:
        collate_fn = DATA_COLLATOR['Default'](**collate_param)
    else:
        collate_fn = DATA_COLLATOR[data_collate_type](**collate_param)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=get_seed_worker(seed),
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def prepare_sampler_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    consumed_samples=0,
    data_sharding=False,
    sampler_type="LengthGroupedSampler",
    group_data=False,
    gradient_accumulation_size=1,
    encoder_dp_size=1,
    initial_global_step_for_sampler=0,
    collate_param=None,
    **kwargs,
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.

    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    process_group = process_group if process_group is not None else _get_default_group()
    if sampler_type == "LengthGroupedSampler":
        if group_data:
            sampler = LengthGroupedSampler(
                batch_size,
                world_size=process_group.size(),
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                gradient_accumulation_size=gradient_accumulation_size,
                encoder_dp_size=encoder_dp_size,
                initial_global_step=initial_global_step_for_sampler,
                lengths=dataset.lengths,
                group_data=group_data,
            )
        else:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=shuffle,
                consumed_samples=consumed_samples,
            )

        if collate_param is None:
            raise ValueError("collate_param must be provided.")
        data_collate_type = collate_param.pop("model_name", None)
        if data_collate_type is None:
            collate_fn = DATA_COLLATOR['Default'](**collate_param)
        else:
            collate_fn = DATA_COLLATOR[data_collate_type](**collate_param)

        return DataLoader(
            dataset,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler if sampler is not None else None,
            drop_last=drop_last,
        )
    
    elif sampler_type == "StatefulDistributedSampler":
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            consumed_samples=consumed_samples,
        )

        if collate_param is None:
            raise ValueError("collate_param must be provided.")
        data_collate_type = collate_param.pop("model_name")
        if data_collate_type is None:
            collate_fn = DATA_COLLATOR['Default'](**collate_param)
        else:
            collate_fn = DATA_COLLATOR[data_collate_type](**collate_param)

        return DataLoader(
            dataset,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            worker_init_fn=get_seed_worker(seed),
            num_workers=num_workers,
            batch_sampler=batch_sampler,
        )
    else:
        raise NotImplementedError(f"sampler type: {sampler_type}")
