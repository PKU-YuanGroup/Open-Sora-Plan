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

from mindspeed_mm.data.datasets.t2v_dataset import DynamicVideoTextDataset
from mindspeed_mm.data.data_utils.bucket import Bucket
from mindspeed_mm.data.data_utils.aspect_ratio import get_num_pixels
from mindspeed_mm.data.data_utils.utils import format_numel_str



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
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


class BaseRandomBatchSampler(DistributedSampler):
    """
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. Default: ``True``. (It is not implemented that the drop_last is false.)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        consumed_samples: int = 0,
        data_sharding: bool = False,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.total_samples = len(dataset)
        self.micro_batch_size = batch_size
        self.consumed_samples = consumed_samples
        self.data_sharding = data_sharding
        self.epoch = 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * self.num_replicas
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        if not drop_last:
            raise ValueError("It is not implemented that the drop_last is false.")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.num_replicas
            start_idx = self.rank * bucket_size
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_bucket = torch.randperm(bucket_size, generator=g).tolist()
            else:
                idx_range_bucket = list(range(bucket_size))
            idx_range = [start_idx + x for x in idx_range_bucket[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_total = \
                    torch.randperm(full_bucket_size, generator=g).tolist()
            else:
                idx_range_total = list(range(full_bucket_size))
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.rank::self.num_replicas]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket,
    )


class VariableVideoBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: DynamicVideoTextDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        for resolution, configs in bucket_config.items():
            bucket_config[resolution] = {int(k):tuple(v) for k, v in configs.items()}
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None

        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

    def __iter__(self) -> Iterator[List[int]]:
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas : (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bucket.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // dist.get_world_size()

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()

        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        logging.info("Building buckets...")
        bucket_ids = self.dataset.data_samples.parallel_apply(
            apply,
            axis=1,
            method=self.bucket.get_bucket_id,
            frame_interval=self.dataset.frame_interval,
            seed=self.seed + self.epoch,
            num_bucket=self.bucket.num_bucket,
        )

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        bucket_sample_dict = self.group_by_bucket()
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if dist.get_rank() == 0 and self.verbose:
            logging.info("Bucket Info:")
            logging.info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            logging.info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            logging.info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            logging.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps * self.num_replicas}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)