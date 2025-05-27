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
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from mindspeed_mm.data.data_utils.utils import get_seed_worker, collate_fn_default
from mindspeed_mm.data.datasets.t2v_dataset import DynamicVideoTextDataset
from mindspeed_mm.data.dataloader.sampler import (
    LengthGroupedSampler,
    StatefulDistributedSampler,
    VariableVideoBatchSampler,
    BaseRandomBatchSampler,
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
    
    elif sampler_type == "BaseRandomBatchSampler":
        batch_sampler = BaseRandomBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            consumed_samples=consumed_samples,
            data_sharding=data_sharding,
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
    elif sampler_type == "SequentialSampler":
        return build_sequential_loader(DataLoaderArgs(dataset,
                                                      batch_size,
                                                      drop_last,
                                                      pin_memory,
                                                      process_group,
                                                      num_workers))
    else:
        raise NotImplementedError(f"sampler type: {sampler_type}")


class DistributedBatchSampler(BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """
    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False, gradient_accumulation_steps=None):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            raise ValueError('please select rank')
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.effective_batch_size = batch_size if gradient_accumulation_steps is None else batch_size * gradient_accumulation_steps

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter * self.effective_batch_size:
                    yield tbatch
                    self.start_iter = 0
                i += len(batch)
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= self.batch_size
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        if start >= len(batch):
            return batch[0:1]
        else:
            return batch[start:end]


@dataclass
class DataLoaderArgs:
    dataset: object
    batch_size: int
    drop_last: bool
    pin_memory: bool
    process_group: object
    num_workers: int


def build_sequential_loader(args: DataLoaderArgs):
    sampler = SequentialSampler(args.dataset)

    world_size = torch.distributed.get_world_size(group=args.process_group)
    rank = args.process_group.rank()
    distributed = world_size > 1
    batch_size = args.batch_size * world_size

    if distributed:
        batch_sampler = DistributedBatchSampler(sampler,
                                                batch_size,
                                                args.drop_last,
                                                rank,
                                                world_size,
                                                gradient_accumulation_steps=1)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, args.drop_last)

    data_loader = torch.utils.data.DataLoader(args.dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=args.pin_memory,
                                              collate_fn=None,
                                              prefetch_factor=4 if args.num_workers > 0 else None)
    return data_loader

