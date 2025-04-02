from typing import Iterator, List, Optional
import math
import logging
import random
from collections import Counter, OrderedDict, defaultdict
from pprint import pformat

import torch
import torch.nn.functional as F

from mindspeed_mm.data.data_utils.constants import (
    PROMPT_IDS, 
    PROMPT_MASK, 
    VIDEO, 
    VIDEO_MASK,
    PROMPT_IDS_2,
    PROMPT_MASK_2
)



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
        group_data: bool = False,
        max_height: int = 480,
        max_width: int = 640,
        ae_stride: int = 8,
        ae_stride_t: int = 4,
        patch_size: int = 2,
        patch_size_t: int = 1,
        num_frames: int = 13,
    ):
        self.batch_size = batch_size
        self.group_data = group_data

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
        input_ids_2 = [i[PROMPT_IDS_2] for i in batch]  # b [1 l]
        cond_mask_2 = [i[PROMPT_MASK_2] for i in batch]  # b [1 l]
        if all([i is None for i in input_ids_2]):
            input_ids_2 = None
        if all([i is None for i in cond_mask_2]):
            cond_mask_2 = None
        return batch_tubes, input_ids, cond_mask, input_ids_2, cond_mask_2

    def __call__(self, batch):
        batch_tubes, input_ids, cond_mask, input_ids_2, cond_mask_2 = self.package(batch)

        ds_stride = self.ae_stride * self.patch_size
        t_ds_stride = self.ae_stride_t * self.patch_size_t

        pad_batch_tubes, attention_mask, input_ids, cond_mask, input_ids_2, cond_mask_2 = self.process(
            batch_tubes,
            input_ids,
            cond_mask,
            input_ids_2,
            cond_mask_2,
            t_ds_stride,
            ds_stride,
            self.max_thw,
            self.ae_stride_thw,
        )
        if torch.any(torch.isnan(pad_batch_tubes)):
            raise AssertionError("after pad_batch_tubes.")
        return {
            VIDEO: pad_batch_tubes,
            VIDEO_MASK: attention_mask,
            PROMPT_IDS: input_ids,
            PROMPT_MASK: cond_mask,
            PROMPT_IDS_2: input_ids_2,
            PROMPT_MASK_2: cond_mask_2,
        }

    def process(
        self,
        batch_tubes,
        input_ids,
        cond_mask,
        input_ids_2, 
        cond_mask_2, 
        t_ds_stride,
        ds_stride,
        max_thw,
        ae_stride_thw,
    ):
        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]  # [(c t h w), (c t h w)]
        if len(batch_input_size) != self.batch_size:
            raise AssertionError("batch_input_size and batch_size are not equal.")
        
        is_grouped = self.group_data or self.batch_size == 1
        if is_grouped:  
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
                # print(
                #     batch_input_size,
                #     idx_length_dict,
                #     count_dict,
                #     sorted_by_value,
                #     pick_length,
                #     candidate_batch,
                #     random_select_batch,
                # )
                pick_idx = candidate_batch + random_select_batch

                batch_tubes = [batch_tubes[i] for i in pick_idx]
                batch_input_size = [
                    i.shape for i in batch_tubes
                ]  # [(c t h w), (c t h w)]
                input_ids = [input_ids[i] for i in pick_idx]  # b [1, l]
                cond_mask = [cond_mask[i] for i in pick_idx]  # b [1, l]
                if input_ids_2 is not None:
                    input_ids_2 = [input_ids_2[i] for i in pick_idx]  # b [1, l]
                if cond_mask_2 is not None:
                    cond_mask_2 = [cond_mask_2[i] for i in pick_idx]  # b [1, l]

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
        if is_grouped:
            if not torch.all(attention_mask.bool()):
                raise AssertionError("All elements of attention_mask are zero")

        input_ids = torch.stack(input_ids)  # b 1 l
        cond_mask = torch.stack(cond_mask)  # b 1 l
        input_ids_2 = torch.stack(input_ids_2) if input_ids_2 is not None else input_ids_2  # b 1 l
        cond_mask_2 = torch.stack(cond_mask_2) if cond_mask_2 is not None else cond_mask_2  # b 1 l

        return (pad_batch_tubes, attention_mask, input_ids, cond_mask, input_ids_2, cond_mask_2)

DATA_COLLATOR = {
    "Default": Collate,
    "GroupLength": Collate,
}
