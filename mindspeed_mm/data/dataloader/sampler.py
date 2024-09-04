import math
import random
import warnings
from collections import Counter, OrderedDict, defaultdict
from pprint import pprint
from typing import Iterator, List, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from mindspeed_mm.data.data_utils.constants import PROMPT_IDS, PROMPT_MASK, VIDEO
from mindspeed_mm.data.datasets.t2v_dataset import VariableT2VDataset


def pad_to_multiple(number, ds_stride):
    remainder = number % ds_stride
    if remainder == 0:
        return number
    else:
        padding = ds_stride - remainder
        return number + padding


class Collate:
    """
    Provide the parameter (collate_fn) to the dataloader
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_frames: int = 13,
        group_frame: bool = False,
        group_resolution: bool = False,
        max_height: int = 480,
        max_width: int = 640,
        ae_stride: int = 8,
        ae_stride_t: int = 4,
        patch_size: int = 2,
        patch_size_t: int = 1,
    ):
        self.batch_size = batch_size
        self.group_frame = group_frame
        self.group_resolution = group_resolution

        self.max_height = max_height
        self.max_width = max_width
        self.ae_stride = ae_stride

        self.ae_stride_t = ae_stride_t
        self.ae_stride_thw = (self.ae_stride_t, self.ae_stride, self.ae_stride)

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        self.num_frames = num_frames
        self.max_thw = (self.num_frames, self.max_height, self.max_width)

    def package(self, batch):
        batch_tubes = [i[VIDEO] for i in batch]  # b [c t h w]
        input_ids = [i[PROMPT_IDS] for i in batch]  # b [1 l]
        cond_mask = [i[PROMPT_MASK] for i in batch]  # b [1 l]
        return batch_tubes, input_ids, cond_mask

    def __call__(self, batch):
        batch_tubes, input_ids, cond_mask = self.package(batch)

        ds_stride = self.ae_stride * self.patch_size
        t_ds_stride = self.ae_stride_t * self.patch_size_t

        pad_batch_tubes, attention_mask, input_ids, cond_mask = self.process(
            batch_tubes,
            input_ids,
            cond_mask,
            t_ds_stride,
            ds_stride,
            self.max_thw,
            self.ae_stride_thw,
        )
        if torch.any(torch.isnan(pad_batch_tubes)):
            raise AssertionError("after pad_batch_tubes.")
        return pad_batch_tubes, attention_mask, input_ids, cond_mask

    def process(
        self,
        batch_tubes,
        input_ids,
        cond_mask,
        t_ds_stride,
        ds_stride,
        max_thw,
        ae_stride_thw,
    ):
        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]  # [(c t h w), (c t h w)]
        if len(batch_input_size) != self.batch_size:
            raise AssertionError("batch_input_size and batch_size are not equal.")
        if self.group_frame or self.group_resolution or self.batch_size == 1:  #
            len_each_batch = batch_input_size
            idx_length_dict = dict([*zip(list(range(self.batch_size)), len_each_batch)])
            count_dict = Counter(len_each_batch)
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
                    for _ in range(len(len_each_batch) - len(candidate_batch))
                ]
                print(
                    batch_input_size,
                    idx_length_dict,
                    count_dict,
                    sorted_by_value,
                    pick_length,
                    candidate_batch,
                    random_select_batch,
                )
                pick_idx = candidate_batch + random_select_batch

                batch_tubes = [batch_tubes[i] for i in pick_idx]
                batch_input_size = [
                    i.shape for i in batch_tubes
                ]  # [(c t h w), (c t h w)]
                input_ids = [input_ids[i] for i in pick_idx]  # b [1, l]
                cond_mask = [cond_mask[i] for i in pick_idx]  # b [1, l]

            for i in range(1, self.batch_size):
                if batch_input_size[0] != batch_input_size[i]:
                    raise AssertionError(
                        f"batch_input_size{0} and batch_input_size{i} are not equal."
                    )
            max_t = max([i[1] for i in batch_input_size])
            max_h = max([i[2] for i in batch_input_size])
            max_w = max([i[3] for i in batch_input_size])
        else:
            max_t, max_h, max_w = max_thw
        pad_max_t, pad_max_h, pad_max_w = (
            pad_to_multiple(max_t - 1 + self.ae_stride_t, t_ds_stride),
            pad_to_multiple(max_h, ds_stride),
            pad_to_multiple(max_w, ds_stride),
        )
        pad_max_t = pad_max_t + 1 - self.ae_stride_t
        each_pad_t_h_w = [
            [pad_max_t - i.shape[1], pad_max_h - i.shape[2], pad_max_w - i.shape[3]]
            for i in batch_tubes
        ]
        pad_batch_tubes = [
            F.pad(im, (0, pad_w, 0, pad_h, 0, pad_t), value=0)
            for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)
        ]
        pad_batch_tubes = torch.stack(pad_batch_tubes, dim=0)

        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [
            ((max_tube_size[0] - 1) // ae_stride_thw[0] + 1),
            max_tube_size[1] // ae_stride_thw[1],
            max_tube_size[2] // ae_stride_thw[2],
        ]
        valid_latent_size = [
            [
                int(math.ceil((i[1] - 1) / ae_stride_thw[0])) + 1,
                int(math.ceil(i[2] / ae_stride_thw[1])),
                int(math.ceil(i[3] / ae_stride_thw[2])),
            ]
            for i in batch_input_size
        ]
        attention_mask = [
            F.pad(
                torch.ones(i, dtype=pad_batch_tubes.dtype),
                (
                    0,
                    max_latent_size[2] - i[2],
                    0,
                    max_latent_size[1] - i[1],
                    0,
                    max_latent_size[0] - i[0],
                ),
                value=0,
            )
            for i in valid_latent_size
        ]
        attention_mask = torch.stack(attention_mask)  # b t h w
        if self.batch_size == 1 or self.group_frame or self.group_resolution:
            if not torch.all(attention_mask.bool()):
                raise AssertionError("All elements of attention_mask are zero")

        input_ids = torch.stack(input_ids)  # b 1 l
        cond_mask = torch.stack(cond_mask)  # b 1 l

        return pad_batch_tubes, attention_mask, input_ids, cond_mask


def split_to_even_chunks(indices, lengths, num_chunks, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        chunks = [indices[i::num_chunks] for i in range(num_chunks)]
    else:
        num_indices_per_chunk = len(indices) // num_chunks

        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        for index in indices:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float("inf")

    pad_chunks = []
    for idx, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            if batch_size <= len(chunk):
                raise AssertionError(
                    "The batch_size must be larger than the length of chunk."
                )

            if len(chunk) != 0:
                chunk = chunk + [
                    random.choice(chunk) for _ in range(batch_size - len(chunk))
                ]
            else:
                chunk = random.choice(pad_chunks)
                print(chunks[idx], "->", chunk)
        pad_chunks.append(chunk)
    return pad_chunks


def group_frame_fun(indices, lengths):
    # sort by num_frames
    indices.sort(key=lambda i: lengths[i], reverse=True)
    return indices


def group_resolution_fun(indices):
    raise NotImplementedError


def group_frame_and_resolution_fun(indices):
    raise NotImplementedError


def last_group_frame_fun(shuffled_megabatches, lengths):
    re_shuffled_megabatches = []
    for i_megabatch, megabatch in enumerate(shuffled_megabatches):
        re_megabatch = []
        for i_batch, batch in enumerate(megabatch):
            if len(batch) == 0:
                raise AssertionError("The length of batch is zero")
            len_each_batch = [lengths[i] for i in batch]
            idx_length_dict = dict([*zip(batch, len_each_batch)])
            count_dict = Counter(len_each_batch)
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


def last_group_resolution_fun(indices):
    raise NotImplementedError


def last_group_frame_and_resolution_fun(indices):
    raise NotImplementedError


def get_length_grouped_indices(
    lengths,
    batch_size,
    world_size,
    generator=None,
    group_frame=False,
    group_resolution=False,
    seed=42,
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(
            seed
        )  # every rank will generate a fixed order but random index

    indices = torch.randperm(len(lengths), generator=generator).tolist()

    if group_frame and not group_resolution:
        indices = group_frame_fun(indices, lengths)
    elif not group_frame and group_resolution:
        indices = group_resolution_fun(indices)
    elif group_frame and group_resolution:
        indices = group_frame_and_resolution_fun(indices)

    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size, batch_size)
        for megabatch in megabatches
    ]

    indices = torch.randperm(len(megabatches), generator=generator).tolist()
    shuffled_megabatches = [megabatches[i] for i in indices]
    if group_frame and not group_resolution:
        shuffled_megabatches = last_group_frame_fun(shuffled_megabatches, lengths)
    elif not group_frame and group_resolution:
        shuffled_megabatches = last_group_resolution_fun(shuffled_megabatches, indices)
    elif group_frame and group_resolution:
        shuffled_megabatches = last_group_frame_and_resolution_fun(
            shuffled_megabatches, indices
        )

    out_list = []
    for megabatch in shuffled_megabatches:
        for batch in megabatch:
            for i in batch:
                out_list.append(i)
    return out_list


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        group_frame=False,
        group_resolution=False,
        generator=None,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        if world_size == -1:
            raise ValueError("world_size must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.group_frame = group_frame
        self.group_resolution = group_resolution
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(
            self.lengths,
            self.batch_size,
            self.world_size,
            group_frame=self.group_frame,
            group_resolution=self.group_resolution,
            generator=self.generator,
        )
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


# TODO
class VariableVideoBatchSampler(DistributedSampler):
    pass
