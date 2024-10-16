from typing import Optional

from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from mindspeed_mm.data.data_utils.utils import get_seed_worker, collate_fn_default
from mindspeed_mm.data.datasets.t2v_dataset import DynamicVideoTextDataset
from mindspeed_mm.data.dataloader.sampler import (
    Collate,
    LengthGroupedSampler,
    StatefulDistributedSampler,
    VariableVideoBatchSampler,
    BaseRandomBatchSampler,
)
from mindspeed_mm.data.dataloader.data_collator import DATA_COLLATOR


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
    if collate_param:
        data_collate_type = collate_param.pop("model_name")
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
    sampler_type="stateful_distributed_sampler",
    group_frame=False,
    group_resolution=False,
    world_size=-1,
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
    if sampler_type == "stateful_distributed_sampler":
        collate_fn = None
        if collate_param:
            data_collate_type = collate_param.pop("model_name")
            collate_fn = DATA_COLLATOR[data_collate_type](**collate_param)

        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=get_seed_worker(seed),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    elif sampler_type == "LengthGroupedSampler":
        sampler = (
            LengthGroupedSampler(
                batch_size,
                world_size=world_size,
                lengths=dataset.lengths,
                group_frame=group_frame,
                group_resolution=group_resolution,
            )
            if (group_frame or group_resolution)
            else None
        )
        if sampler is None:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=shuffle,
            )

        if collate_param is None:
            raise ValueError("collate_param must be provided.")
        collate_fn = Collate(**collate_param)

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
        collate_fn = Collate(**collate_param)

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


def prepare_variable_dataloader(
        dataset,
        shuffle=False,
        seed=1024,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        process_group: Optional[ProcessGroup] = None,
        bucket_config=None,
        num_bucket_build_workers=1,
        sampler_type="variable_video_batch_sampler",
        **kwargs,
    ):
    if isinstance(dataset, DynamicVideoTextDataset) and sampler_type == "variable_video_batch_sampler":
        batch_sampler = VariableVideoBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
        )

        return DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    worker_init_fn=get_seed_worker(seed),
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    collate_fn=collate_fn_default,
                    **kwargs,
                )
    else:
        return NotImplementedError
