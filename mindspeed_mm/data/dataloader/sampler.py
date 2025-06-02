from typing import Iterator, List, Optional
import math
import logging
import random
from collections import Counter, OrderedDict, defaultdict
from pprint import pformat
from pandarallel import pandarallel

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from megatron.legacy.data.data_samplers import RandomSeedDataset


def split_to_even_chunks(megabatch, lengths, world_size, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # batch_size=2, world_size=2
    # [1, 2, 3, 4] -> [[1, 2], [3, 4]]
    # [1, 2, 3] -> [[1, 2], [3]]
    # [1, 2] -> [[1], [2]]
    # [1] -> [[1], []]
    chunks = [megabatch[i::world_size] for i in range(world_size)]

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):  
            if batch_size <= len(chunk):
                raise AssertionError("batch_size must greater than len_chunk !")
            if len(chunk) != 0:  # [[1, 2], [3]] -> [[1, 2], [3, 3]]
                chunk = chunk + [random.choice(chunk) for _ in range(batch_size - len(chunk))]
            else:
                chunk = random.choice(pad_chunks)  # [[1], []] -> [[1], [1]]
                print(chunks[idx], '->', chunk)
        pad_chunks.append(chunk)
    return pad_chunks


def last_group_data_fun(shuffled_megabatches, lengths):
    re_shuffled_megabatches = []
    for i_megabatch, megabatch in enumerate(shuffled_megabatches):
        re_megabatch = []
        for i_batch, batch in enumerate(megabatch):
            if len(batch) == 0:
                raise AssertionError("The length of batch is zero")
            len_each_batch = [lengths[i] for i in batch]
            idx_length_dict = dict([*zip(batch, len_each_batch)])
            count_dict = Counter(len_each_batch)
            # This means that there are multiple different shapes of data on a certain GPU
            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [
                    idx
                    for idx, length in idx_length_dict.items()
                    if length == pick_length
                ]
                random_select_batch = [
                    random.choice(candidate_batch)
                    for i in range(len(len_each_batch) - len(candidate_batch))
                ]
                batch = candidate_batch + random_select_batch
            re_megabatch.append(batch)
        re_shuffled_megabatches.append(re_megabatch)
    return re_shuffled_megabatches


def group_data_fun(lengths, generator=None):
    # counter is decrease order
    counter = Counter(lengths)  # counter {'1x256x256': 3, ''}   lengths ['1x256x256', '1x256x256', '1x256x256', ...]
    grouped_indices = defaultdict(list)
    for idx, item in enumerate(lengths):  # group idx to a list
        grouped_indices[item].append(idx)

    grouped_indices = dict(grouped_indices)  # {'1x256x256': [0, 1, 2], ...}
    sorted_indices = [grouped_indices[item] for (item, _) in sorted(counter.items(), key=lambda x: x[1], reverse=True)]
    
    # shuffle in each group
    shuffle_sorted_indices = []
    for indice in sorted_indices:
        shuffle_idx = torch.randperm(len(indice), generator=generator).tolist()
        shuffle_sorted_indices.extend([indice[idx] for idx in shuffle_idx])
    return shuffle_sorted_indices


def get_length_grouped_data_indices(
        lengths, 
        batch_size, 
        world_size, 
        gradient_accumulation_size=1, 
        encoder_dp_size=1,
        initial_global_step=0, 
        generator=None, 
        group_data=False, 
        seed=42
    ):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(seed)

    if group_data:
        indices = group_data_fun(lengths, generator)
    else:
        indices = torch.randperm(len(lengths), generator=generator).tolist()
    
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]

    # Ensure that the sample quantity for each GPU is consistent, and only a small portion of GPU data shapes do not match
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size, batch_size) for megabatch in megabatches]

    indices_mega = torch.randperm(len(megabatches), generator=generator).tolist()

    shuffled_megabatches = [megabatches[i] for i in indices_mega]

    if group_data:
        shuffled_megabatches = last_group_data_fun(shuffled_megabatches, lengths)
    
    initial_global_step = initial_global_step * gradient_accumulation_size // encoder_dp_size
    print(f"initial_global_step: {initial_global_step}, gradient_accumulation_size: {gradient_accumulation_size}, encoder_dp_size: {encoder_dp_size}")
    shuffled_megabatches = shuffled_megabatches[initial_global_step:]

    out_list = []
    for megabatch in shuffled_megabatches:
        for batch in megabatch:
            for i in batch:
                out_list.append(i)
    return out_list


class LengthGroupedSampler(DistributedSampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        gradient_accumulation_size: int = 1, 
        initial_global_step: int = 0, 
        encoder_dp_size=1,
        lengths: Optional[List[int]] = None,
        group_data=False,
        generator=None,
    ):
        super().__init__(dataset=lengths, num_replicas=num_replicas, rank=rank)

        if lengths is None:
            raise ValueError("Lengths must be provided.")
        if world_size == -1:
            raise ValueError("world_size must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.initial_global_step = initial_global_step
        self.gradient_accumulation_size = gradient_accumulation_size
        self.encoder_dp_size = encoder_dp_size
        self.lengths = lengths
        self.group_data = group_data
        self.generator = generator

        print('self.lengths, self.initial_global_step, self.batch_size, self.world_size, self.gradient_accumulation_size, self.encoder_dp_size,', 
            len(self.lengths), self.initial_global_step, self.batch_size, self.world_size, self.gradient_accumulation_size, self.encoder_dp_size)

    def __len__(self):
        if self.group_data:
            return len(self.lengths) - self.initial_global_step * self.batch_size * self.world_size * self.gradient_accumulation_size // self.encoder_dp_size
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_data_indices(
            self.lengths,
            self.batch_size,
            self.world_size,
            self.gradient_accumulation_size,
            self.encoder_dp_size,
            self.initial_global_step,
            group_data=self.group_data,
            generator=self.generator,
        )
        indices = indices[self.rank:len(indices):self.num_replicas]
        return iter(indices)

class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        consumed_samples: int = 0,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0
        self.consumed_samples = consumed_samples // num_replicas

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        self.start_index = self.consumed_samples % self.num_samples
        indices = indices[self.start_index:]
        actual_indices_len = len(indices)
        self.consumed_samples += actual_indices_len
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index

