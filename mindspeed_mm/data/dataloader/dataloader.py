from typing import Iterator, Optional

import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Dataset

from mindspeed_mm.data.data_utils.utils import get_seed_worker
from mindspeed_mm.data.dataloader.sampler import StatefulDistributedSampler


def prepare_base_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=get_seed_worker(seed),
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
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
    sampler_type="stateful_distributed_sampler",
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
    if sampler_type == "stateful_distributed_sampler":
        process_group = process_group or _get_default_group()
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            worker_init_fn=get_seed_worker(seed),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
    else:
        raise NotImplementedError(f"sampler type: {sampler_type}")


# TODO
def prepare_variable_dataloader():
    pass
