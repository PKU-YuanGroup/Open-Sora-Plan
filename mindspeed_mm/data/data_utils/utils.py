# Modified from huggingface diffusers repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# TextProcesser: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py


import os
import re
import gc
import cv2
import sys
import html
import math
import copy
import random
import urllib.parse as ul
from tqdm import tqdm
from fractions import Fraction
from collections import Counter
from typing import Any, Dict, Optional, Tuple, Union, Sequence

try:
    import decord
except ImportError:
    print("Failed to import decord module.")

import av
import ftfy
import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as TT
from PIL import Image
from bs4 import BeautifulSoup
from einops import rearrange
from torchvision import get_video_backend
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from torchvision.io.video import (
    _align_audio_frames,
    _check_av_available,
    _log_api_usage_once,
    _read_from_stream,
    _video_opt,
)
import transformers
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers.trainer_pt_utils import LabelSmoother
from packaging import version
import tokenizers

from mindspeed_mm.data.data_utils.data_transform import (
    TemporalRandomCrop, 
    Expand2Square,
    get_params,
    calculate_statistics,
    maxhwresize
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class DataFileReader:
    """get the data from different types of files such as csv/json/parquat"""

    def __init__(self, data_storage_mode="standard"):
        self.data_storage_mode = data_storage_mode

    def __call__(self, data_path, return_type="list"):
        if self.data_storage_mode == "standard":
            return self.get_datasamples(data_path, return_type=return_type)
        elif self.data_storage_mode == "combine":
            return self.get_cap_list(data_path)
        else:
            raise NotImplementedError("Not support now.")

    def get_datasamples(self, data_path, return_type="list"):
        if data_path.endswith(".json"):
            data_out = pd.read_json(data_path)
        elif data_path.endswith(".pkl"):
            data_out = pd.read_pickle(data_path)
        elif data_path.endswith(".jsonl"):
            data_out = pd.read_json(data_path, lines=True)
        elif data_path.endswith(".parquat"):
            data_out = pd.read_parquat(data_path)
        else:
            raise NotImplementedError(f"Unsupported file format: {data_path}")

        if return_type == "list":
            if isinstance(data_out, pd.DataFrame):
                return data_out.to_dict("records")
            elif isinstance(data_out, list):
                return data_out
        else:
            raise NotImplementedError(f"Unsupported return_type: {return_type}")

    def get_cap_list(self, data_path):
        cap_lists = []
        with open(data_path, "r") as f:
            folder_anno = [
                i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0
            ]
        for folder, anno in folder_anno:
            sub_list = self.get_datasamples(anno)
            print(f"Building {anno}...")
            for sub in sub_list:
                sub["path"] = os.path.join(folder, sub["path"])
            cap_lists += sub_list
        return cap_lists


class DecordDecoder(object):
    def __init__(self, url, num_threads=1):

        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.reader = decord.VideoReader(url,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)

    def get_avg_fps(self):
        return self.reader.get_avg_fps() if self.reader.get_avg_fps() > 0 else 30.0

    def get_num_frames(self):
        return len(self.reader)

    def get_height(self):
        return self.reader[0].shape[0] if self.get_num_frames() > 0 else 0

    def get_width(self):
        return self.reader[0].shape[1] if self.get_num_frames() > 0 else 0

    # output shape [T, H, W, C]
    def get_batch(self, frame_indices):
        try:
            #frame_indices[0] = 1000
            video_data = self.reader.get_batch(frame_indices).asnumpy()
            video_data = torch.from_numpy(video_data)
            return video_data
        except Exception as e:
            print(f"Get_batch execption: {e}")
            return None


class VideoReader:
    """support some methods to read video"""

    def __init__(self, video_reader_type=None):
        self.video_reader_type = video_reader_type
    
    def __call__(self, video_path):
        is_decord_read = False
        info = None

        if self.video_reader_type == "decoder":
            vframes = DecordDecoder(video_path)
            is_decord_read = True
        elif self.video_reader_type == "torchvision":
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, pts_unit="sec", output_format="TCHW"
            )  # [T: temporal, C: channel, H: height, W: width]
        else:
            raise NotImplementedError(
                f"Unsupported video reader type: {self.video_reader_type}"
            )
        return vframes, info, is_decord_read


class VideoProcesser:
    """Used for video data preprocessing"""

    def __init__(
            self,
            num_frames=16,
            frame_interval=1,
            train_pipeline=None,
            data_storage_mode="standard",
            train_fps=24,
            speed_factor=1.0,
            too_long_factor=5.0,
            drop_short_ratio=1.0,
            max_height=480,
            max_width=640,
            max_hxw=None,
            min_hxw=None,
            force_resolution=True,
            force_5_ratio=False,
            seed=42,
            hw_stride=32,
            max_h_div_w_ratio=2.0,
            min_h_div_w_ratio=None,
            ae_stride_t=4,
            sp_size=1,
            train_sp_batch_size=1,
            gradient_accumulation_size=1,
            batch_size=1,
            min_num_frames=29,
            **kwargs,
    ):
        self.num_frames = num_frames
        self.train_pipeline = train_pipeline
        self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline)
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
        self.data_storage_mode = data_storage_mode

        self.max_height = max_height
        self.max_width = max_width

        if self.data_storage_mode == "combine":
            self.train_fps = train_fps
            self.speed_factor = speed_factor
            self.too_long_factor = too_long_factor
            self.drop_short_ratio = drop_short_ratio
            self.max_hxw = max_hxw
            self.min_hxw = min_hxw
            self.force_resolution = force_resolution
            self.force_5_ratio = force_5_ratio
            self.seed = seed
            self.generator = torch.Generator().manual_seed(self.seed) 
            self.hw_stride = hw_stride
            self.max_h_div_w_ratio = max_h_div_w_ratio
            self.min_h_div_w_ratio = min_h_div_w_ratio if min_h_div_w_ratio is not None else 1 / max_h_div_w_ratio
            self.ae_stride_t = ae_stride_t
            self.sp_size = sp_size
            self.train_sp_batch_size = train_sp_batch_size
            self.gradient_accumulation_size = gradient_accumulation_size
            self.batch_size = batch_size
            self.min_num_frames = min_num_frames


    def __call__(
        self, 
        vframes, 
        start_frame_idx,
        clip_total_frames,
        predefine_frame_indice=None,
        fps=16,
        is_decord_read=False, 
        crop=[None, None, None, None]
    ):
        
        if self.data_storage_mode == "combine":
            video = self.combine_data_video_process(
                vframes,
                start_frame_idx,
                clip_total_frames,
                is_decord_read=is_decord_read,
                predefine_frame_indice=predefine_frame_indice,
                fps=fps,
                crop=crop,
            )
        return video

    def get_batched_data(self, vframes, frame_indices, crop=[None, None, None, None]):
        video_data = vframes.get_batch(frame_indices)
        try:
            s_x, e_x, s_y, e_y = crop
        except:
            s_x, e_x, s_y, e_y = None, None, None, None
        if video_data is not None:
            video_data = video_data.permute(0, 3, 1, 2)  # (T H W C) -> (T C H W)
            if s_y is not None:
                video_data = video_data[:, :, s_y: e_y, s_x: e_x]
        else:
            raise ValueError(f'Get video_data {video_data}')
        return video_data

    def combine_data_video_process(
        self, vframes, start_frame_idx, clip_total_frames, is_decord_read=True, predefine_frame_indice=None, fps=16, crop=[None, None, None, None]
    ):
        predefine_num_frames = len(predefine_frame_indice)

        # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
        frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        frame_indices = np.arange(start_frame_idx, start_frame_idx + clip_total_frames, frame_interval).astype(int)
        frame_indices = frame_indices[frame_indices < start_frame_idx + clip_total_frames]
        # speed up
        max_speed_factor = len(frame_indices) / self.num_frames
        if self.speed_factor > 1 and max_speed_factor > 1:
            # speed_factor = random.uniform(1.0, min(self.speed_factor, max_speed_factor))
            speed_factor = min(self.speed_factor, max_speed_factor)
            target_frame_count = int(len(frame_indices) / speed_factor)
            speed_frame_idx = np.linspace(0, len(frame_indices) - 1, target_frame_count, dtype=int)
            frame_indices = frame_indices[speed_frame_idx]

        #  too long video will be temporal-crop randomly
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index: end_index]
            # frame_indices = frame_indices[:self.num_frames]  # head crop

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
        )
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(f'video has {clip_total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        frame_indices = frame_indices[:end_frame_idx]
        if predefine_num_frames != len(frame_indices):
            raise ValueError(f'video predefine_num_frames ({predefine_num_frames}) ({predefine_frame_indice}) is not equal with frame_indices ({len(frame_indices)}) ({frame_indices})')
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(f'video has {clip_total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        video = self.get_batched_data(vframes, frame_indices, crop)
        # TCHW -> TCHW
        video = self.video_transforms(video)
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return video

    def define_frame_index(self, cap_list):
        shape_idx_dict = {}
        new_cap_list = []
        sample_size = []
        aesthetic_score = []
        cnt_vid = 0
        cnt_img = 0
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_no_aesthetic = 0
        cnt_img_res_mismatch_stride = 0
        cnt_vid_res_mismatch_stride = 0
        cnt_img_aspect_mismatch = 0
        cnt_vid_aspect_mismatch = 0
        cnt_img_res_too_small = 0
        cnt_vid_res_too_small = 0
        cnt_vid_after_filter = 0
        cnt_img_after_filter = 0
        cnt_fps_too_low = 0
        cnt = len(cap_list)

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        total_batch_size = self.batch_size * world_size * self.gradient_accumulation_size
        total_batch_size = total_batch_size // self.sp_size * self.train_sp_batch_size
        # discard samples with shape which num is less than 4 * total_batch_size
        filter_major_num = 4 * total_batch_size

        for i in tqdm(cap_list, desc=f"flitering samples"):
            path = i["path"]

            if path.endswith('.mp4'):
                cnt_vid += 1
            elif path.endswith('.jpg'):
                cnt_img += 1

            # ======no aesthetic=====
            if i.get("aesthetic", None) is None or i.get("aes", None) is None:
                cnt_no_aesthetic += 1
            else:
                aesthetic_score.append(i.get("aesthetic", None) or i.get("aes", None))

            cap = i.get("cap", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue

            # ======resolution mismatch=====
            if i.get("resolution", None) is None:
                cnt_no_resolution += 1
                continue
            else:
                if i["resolution"].get("height", None) is None or i["resolution"].get("width", None) is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    height, width = i["resolution"]["height"], i["resolution"]["width"]
                    if height <= 0 or width <= 0:
                        cnt_no_resolution += 1
                        continue

                    # filter aspect
                    is_pick = filter_resolution(
                        height, 
                        width, 
                        max_h_div_w_ratio=self.max_h_div_w_ratio, 
                        min_h_div_w_ratio=self.min_h_div_w_ratio
                    )

                    if not is_pick:
                        if path.endswith('.mp4'):
                            cnt_vid_aspect_mismatch += 1
                        elif path.endswith('.jpg'):
                            cnt_img_aspect_mismatch += 1
                        continue

                    # filter min_hxw
                    if height * width < self.min_hxw:
                        if path.endswith('.mp4'):
                            cnt_vid_res_too_small += 1
                        elif path.endswith('.jpg'):
                            cnt_img_res_too_small += 1
                        continue
                    
                    if not self.force_resolution:
                        tr_h, tr_w = maxhwresize(height, width, self.max_hxw, force_5_ratio=self.force_5_ratio)
                        _, _, sample_h, sample_w = get_params(tr_h, tr_w, self.hw_stride, force_5_ratio=self.force_5_ratio)

                        if sample_h <= 0 or sample_w <= 0:
                            if path.endswith('.mp4'):
                                cnt_vid_res_mismatch_stride += 1
                            elif path.endswith('.jpg'):
                                cnt_img_res_mismatch_stride += 1
                            continue
                        # filter min_hxw
                        if sample_h * sample_w < self.min_hxw:
                            if path.endswith('.mp4'):
                                cnt_vid_res_too_small += 1
                            elif path.endswith('.jpg'):
                                cnt_img_res_too_small += 1
                            continue
                        i["resolution"].update(dict(sample_height=sample_h, sample_width=sample_w))
                    else: 
                        target_h_div_w = self.max_height / self.max_width
                        current_h_div_w = height / width
                        min_hxw_scale = max(current_h_div_w / target_h_div_w, target_h_div_w / current_h_div_w)
                        min_hxw = math.ceil(self.min_hxw * min_hxw_scale)
                        # filter min_hxw
                        if height * width < min_hxw:
                            if path.endswith('.mp4'):
                                cnt_vid_res_too_small += 1
                            elif path.endswith('.jpg'):
                                cnt_img_res_too_small += 1
                            continue
                        sample_h, sample_w = self.max_height, self.max_width
                        i["resolution"].update(dict(sample_height=sample_h, sample_width=sample_w))


            if path.endswith(".mp4"):
                fps = i.get('fps', 24)
                # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration. 
                if i['num_frames'] > self.too_long_factor * (self.num_frames * fps / self.train_fps * self.speed_factor):  # too long video is not suitable for this training stage (self.num_frames)
                    cnt_too_long += 1
                    continue

                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps

                if frame_interval < 1.0:
                    cnt_fps_too_low += 1
                    continue

                start_frame_idx = i.get("cut", [0])[0]
                i["start_frame_idx"] = start_frame_idx
                frame_indices = np.arange(
                    start_frame_idx, start_frame_idx + i["num_frames"], frame_interval
                ).astype(int)
                frame_indices = frame_indices[frame_indices < start_frame_idx + i["num_frames"]]

                # comment out it to enable dynamic frames training
                if (
                    len(frame_indices) < self.num_frames
                    and torch.rand(1, generator=self.generator).item() < self.drop_short_ratio
                ):
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                # to find a suitable end_frame_idx, to ensure we do not need pad video
                end_frame_idx = self.find_closest_y(
                    len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
                )
                if (
                    end_frame_idx == -1
                ):  # too short that can not be encoded exactly by videovae
                    cnt_too_short += 1
                    continue
                frame_indices = frame_indices[:end_frame_idx]

                i["sample_frame_index"] = frame_indices.tolist()
                i["sample_num_frames"] = len(i["sample_frame_index"])

                new_cap_list.append(i)
                cnt_vid_after_filter += 1

            elif path.endswith(".jpg"):  # image
                cnt_img_after_filter += 1

                i["sample_frame_index"] = [0]
                i["sample_num_frames"] = 1
                new_cap_list.append(i)
            else:
                raise NameError(
                    f"Unknown file extention {path.split('.')[-1]}, only support .mp4 for video and .jpg for image"
                )
            
            sample_size.append(f"{len(i['sample_frame_index'])}x{sample_h}x{sample_w}")

        counter = Counter(sample_size)
        counter_cp = counter
        if not self.force_resolution and self.max_hxw is not None and self.min_hxw is not None:
            assert all([np.prod(np.array(k.split('x')[1:]).astype(np.int32)) <= self.max_hxw for k in counter_cp.keys()])
            assert all([np.prod(np.array(k.split('x')[1:]).astype(np.int32)) >= self.min_hxw for k in counter_cp.keys()])

        len_before_filter_major = len(new_cap_list)
        new_cap_list, sample_size = zip(*[[i, j] for i, j in zip(new_cap_list, sample_size) if counter[j] >= filter_major_num])
        for idx, shape in enumerate(sample_size):
            if shape_idx_dict.get(shape, None) is None:
                shape_idx_dict[shape] = [idx]
            else:
                shape_idx_dict[shape].append(idx)
        cnt_filter_minority = len_before_filter_major - len(new_cap_list)
        counter = Counter(sample_size)
        print(f'no_cap: {cnt_no_cap}, no_resolution: {cnt_no_resolution}\n'
            f'too_long: {cnt_too_long}, too_short: {cnt_too_short}, fps_too_low: {cnt_fps_too_low}\n'
            f'cnt_img_res_mismatch_stride: {cnt_img_res_mismatch_stride}, cnt_vid_res_mismatch_stride: {cnt_vid_res_mismatch_stride}\n'
            f'cnt_img_res_too_small: {cnt_img_res_too_small}, cnt_vid_res_too_small: {cnt_vid_res_too_small}\n'
            f'cnt_img_aspect_mismatch: {cnt_img_aspect_mismatch}, cnt_vid_aspect_mismatch: {cnt_vid_aspect_mismatch}\n'
            f'cnt_filter_minority: {cnt_filter_minority}\n'
            f'Counter(sample_size): {counter}\n'
            f'cnt_vid: {cnt_vid}, cnt_vid_after_filter: {cnt_vid_after_filter}, use_ratio: {round(cnt_vid_after_filter/(cnt_vid+1e-6), 5)*100}%\n'
            f'cnt_img: {cnt_img}, cnt_img_after_filter: {cnt_img_after_filter}, use_ratio: {round(cnt_img_after_filter/(cnt_img+1e-6), 5)*100}%\n'
            f'before filter: {cnt}, after filter: {len(new_cap_list)}, use_ratio: {round(len(new_cap_list)/cnt, 5)*100}%')

        if len(aesthetic_score) > 0:
            stats_aesthetic = calculate_statistics(aesthetic_score)
            print(f"before filter: {cnt}, after filter: {len(new_cap_list)}\n"
                f"aesthetic_score: {len(aesthetic_score)}, cnt_no_aesthetic: {cnt_no_aesthetic}\n"
                f"{len([i for i in aesthetic_score if i>=5.75])} > 5.75, 4.5 > {len([i for i in aesthetic_score if i<=4.5])}\n"
                f"Mean: {stats_aesthetic['mean']}, Var: {stats_aesthetic['variance']}, Std: {stats_aesthetic['std_dev']}\n"
                f"Min: {stats_aesthetic['min']}, Max: {stats_aesthetic['max']}")
            
        return pd.DataFrame(new_cap_list), sample_size, shape_idx_dict

    def find_closest_y(self, x, vae_stride_t=4, model_ds_t=1):
        if x < self.min_num_frames:
            return -1  
        for y in range(x, self.min_num_frames - 1, -1):
            if (y - 1) % vae_stride_t == 0 and ((y - 1) // vae_stride_t + 1) % model_ds_t == 0:
                # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
                # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
                # 8, 1: y in [33, 41, 49, 57, 65, 73, 81, 89, 97, 105]
                # 8, 2: y in [41, 57, 73, 89, 105]
                # 8, 4: y in [57, 89]
                # 8, 8: y in [57]
                return y
        return -1

# # TODO: not suported now
# def type_ratio_normalize(mask_type_ratio_dict):
#     for k, v in mask_type_ratio_dict.items():
#         assert v >= 0, f"mask_type_ratio_dict[{k}] should be non-negative, but got {v}"
#     total = sum(mask_type_ratio_dict.values())
#     length = len(mask_type_ratio_dict)
#     if total == 0:
#         return {k: 1.0 / length for k in mask_type_ratio_dict.keys()}
#     return {k: v / total for k, v in mask_type_ratio_dict.items()}

# # TODO: not suported now
# class InpaintVideoProcesser(VideoProcesser):
#     """Used for video inpaint data preprocessing"""

#     def __init__(
#             self,
#             num_frames=16,
#             frame_interval=1,
#             train_pipeline=None,
#             train_resize_pipeline=None,
#             data_storage_mode="standard",
#             train_fps=24,
#             speed_factor=1.0,
#             drop_short_ratio=1.0,
#             max_height=480,
#             max_width=640,
#             min_clear_ratio=0.0,
#             max_clear_ratio=1.0,
#             mask_type_ratio_dict_video=None,
#             **kwargs,
#     ):
#         super().__init__(
#             num_frames=num_frames,
#             frame_interval=frame_interval,
#             train_pipeline=train_pipeline,
#             data_storage_mode=data_storage_mode,
#             train_fps=train_fps,
#             speed_factor=speed_factor,
#             drop_short_ratio=drop_short_ratio,
#             max_height=max_height,
#             max_width=max_width,
#         )

#         self.train_resize_pipeline = train_resize_pipeline

#         self.mask_type_ratio_dict_video = mask_type_ratio_dict_video if mask_type_ratio_dict_video is not None else {'random_temporal': 1.0}
#         self.mask_type_ratio_dict_video = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_video.items()}
#         self.mask_type_ratio_dict_video = type_ratio_normalize(self.mask_type_ratio_dict_video)

#         print(f"mask_type_ratio_dict_video: {self.mask_type_ratio_dict_video}")

#         self.mask_processor = MaskProcessor(
#             max_height=max_height,
#             max_width=max_width,
#             min_clear_ratio=min_clear_ratio,
#             max_clear_ratio=max_clear_ratio,
#         )


#     def __call__(self, vframes, num_frames=None, frame_interval=None, image_size=None, is_decord_read=False,
#                  predefine_num_frames=13):
#         if image_size:
#             self.resize_transforms = get_transforms(is_video=True, train_pipeline=self.train_resize_pipeline,
#                                                    image_size=image_size)
#             self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
#                                                    image_size=image_size)
#         else:
#             self.resize_transforms = get_transforms(is_video=True, train_pipeline=self.train_resize_pipeline)
#             self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline)
#         if self.data_storage_mode == "standard":
#             total_frames = len(vframes)
#             if num_frames:
#                 self.num_frames = num_frames
#                 self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
#             start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
#             if end_frame_ind - start_frame_ind < self.num_frames:
#                 raise AssertionError("the video does not have enough frames.")
#             frame_indice = np.linspace(
#                 start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
#             )
#             if is_decord_read:
#                 video = vframes.get_batch(frame_indice).asnumpy()
#                 video = torch.from_numpy(video)
#                 # THWC -> TCHW,  [T: temporal, C: channel, H: height, W: width]
#                 video = video.permute(0, 3, 1, 2)
#             else:
#                 video = vframes[frame_indice]  # TCHW

#             video = self.resize_transforms(video)
#             inpaint_cond_data = self.mask_processor(video, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
#             mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

#             video = self.video_transforms(video)  # T C H W -> T C H W
#             masked_video = self.video_transforms(masked_video)  # T C H W -> T C H W

#             video = torch.cat([video, masked_video, mask], dim=1)  # T 2C+1 H W

#             # video = self.video_transforms(video)
#             # TCHW -> CTHW
#             video = video.permute(1, 0, 2, 3)
#         else:
#             video = self.combine_data_video_process(
#                 vframes,
#                 is_decord_read=is_decord_read,
#                 predefine_num_frames=predefine_num_frames,
#             )
#         return video


#     def combine_data_video_process(
#             self, vframes, is_decord_read=True, predefine_num_frames=13
#     ):
#         total_frames = len(vframes)
#         fps = vframes.get_avg_fps() if vframes.get_avg_fps() > 0 else 30.0
#         # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
#         frame_interval = (
#             1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
#         )
#         # some special video should be set to a different number
#         start_frame_idx = 0
#         frame_indices = np.arange(start_frame_idx, total_frames, frame_interval).astype(
#             int
#         )
#         frame_indices = frame_indices[frame_indices < total_frames]
#         # speed up
#         max_speed_factor = len(frame_indices) / self.num_frames
#         if self.speed_factor > 1 and max_speed_factor > 1:
#             speed_factor = min(self.speed_factor, max_speed_factor)
#             target_frame_count = int(len(frame_indices) / speed_factor)
#             speed_frame_idx = np.linspace(
#                 0, len(frame_indices) - 1, target_frame_count, dtype=int
#             )
#             frame_indices = frame_indices[speed_frame_idx]

#         #  too long video will be temporal-crop randomly
#         if len(frame_indices) > self.num_frames:
#             begin_index, end_index = self.temporal_sample(len(frame_indices))
#             frame_indices = frame_indices[begin_index:end_index]

#         # to find a suitable end_frame_idx, to ensure we do not need pad video
#         end_frame_idx = self.find_closest_y(
#             len(frame_indices), vae_stride_t=4, model_ds_t=4
#         )
#         if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
#             raise IndexError(
#                 f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
#             )
#         frame_indices = frame_indices[:end_frame_idx]
#         if predefine_num_frames != len(frame_indices):
#             raise ValueError(
#                 f"predefine_num_frames ({predefine_num_frames}) is not equal with frame_indices ({len(frame_indices)})"
#             )
#         if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
#             raise IndexError(
#                 f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
#             )
#         video = vframes.get_batch(frame_indices).asnumpy()
#         video = torch.from_numpy(video)
#         # (T, H, W, C) -> (T C H W)
#         video = video.permute(0, 3, 1, 2)

#         h, w = video.shape[-2:]
#         if h / w > 17 / 16 or h / w < 8 / 16:
#             raise AssertionError(
#                 f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But the video found ratio is {round(h / w, 2)} with the shape of {video.shape}"
#             )

#         video = self.resize_transforms(video)
#         inpaint_cond_data = self.mask_processor(video, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
#         mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

#         video = self.video_transforms(video)  # T C H W -> T C H W
#         masked_video = self.video_transforms(masked_video)  # T C H W -> T C H W

#         # TCHW -> TCHW
#         # video = self.video_transforms(video)
#         # TCHW -> CTHW
#         video = video.permute(1, 0, 2, 3)
#         return video



class ImageProcesser:
    """Used for image data preprocessing"""

    def __init__(
            self,
            num_frames=16,
            train_pipeline=None,
            image_reader_type="torchvision",
            image_processer_type="image2video",
            **kwargs,
    ):
        self.num_frames = num_frames
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline
        )
        self.train_pipeline = train_pipeline
        self.image_reader_type = image_reader_type
        self.image_processer_type = image_processer_type

    def __call__(self, image_path):
        if self.image_processer_type == "image2image":
            image = self.image_to_image(image_path)
        else:
            raise NotImplementedError(
                f"Unsupported image processer type: {self.image_processer_type}"
            )
        return image

    def image_to_image(self, image_path):
        image = self.image_reader(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  # [1 c h w]
        # [1 C H W] -> num_img [1 C H W]
        image = self.image_transforms(image)
        # [1 C H W] -> [C 1 H W]
        image = image.permute(1, 0, 2, 3)
        return image

    def image_reader(self, image_path):
        if self.image_reader_type in ["torchvision", "CLIPImageProcessor"]:
            image = pil_loader(image_path)
        elif self.image_reader_type == "Image":
            image = Image.open(image_path).convert("RGB")  # [h, w, c]
        else:
            raise NotImplementedError(
                f"Unsupported image reader type: {self.image_reader_type}"
            )
        return image

class TextProcesser:
    """Used for text data preprocessing"""

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + "\)"
        + "\("
        + "\]"
        + "\["
        + "\}"
        + "\{"
        + "\|"
        + "\\"
        + "\/"
        + "\*"
        + r"]{1,}"
    )

    def __init__(
            self,
            model_max_length=120,
            tokenizer=None,
            tokenizer_2=None,
            use_clean_caption=True,
            enable_text_preprocessing=True,
            padding_type="max_length",
            support_chinese=False,
            cfg=0.1,
    ):
        self.model_max_length = model_max_length
        self.padding = padding_type
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.use_clean_caption = use_clean_caption
        self.support_chinese = support_chinese
        self.cfg = cfg
        self.enable_text_preprocessing = enable_text_preprocessing

    def __call__(self, texts):
        if self.enable_text_preprocessing:
            texts_info = [
                TextProcesser.text_preprocessing(text, self.use_clean_caption)
                for text in texts
            ]
            texts_info = texts_info if random.random() > self.cfg else [""]
        else:
            texts_info = texts

        text_tokens_and_mask = self.tokenizer(
            texts_info,
            max_length=self.model_max_length,
            padding=self.padding,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_ids = text_tokens_and_mask["input_ids"]
        prompt_mask = text_tokens_and_mask["attention_mask"]
        prompt_ids_2, prompt_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                texts_info,
                max_length=self.tokenizer_2.model_max_length,
                padding=self.padding,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            prompt_ids_2 = text_tokens_and_mask_2['input_ids']  # 1, l
            prompt_mask_2 = text_tokens_and_mask_2['attention_mask']  # 1, l
        return (prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2)

    @staticmethod
    def text_preprocessing(text, use_clean_caption=True, support_chinese=False):
        if use_clean_caption:
            text = TextProcesser.clean_caption(text, support_chinese=support_chinese)
            text = TextProcesser.clean_caption(text, support_chinese=support_chinese)
        else:
            text = text.lower().strip()
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    @staticmethod
    def clean_caption(caption, support_chinese=False):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        if not support_chinese:
            caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # Uniform quotation marks
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            TextProcesser.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = TextProcesser.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

# # NOTE: not supported now
# class InpaintTextProcesser(TextProcesser):

#     def __init__(
#         self,
#         model_max_length=120,
#         tokenizer=None,
#         use_clean_caption=True,
#         enable_text_preprocessing=True,
#         padding_type="max_length",
#         support_chinese=False,
#         cfg=0.1,
#         default_text_ratio=0.5,
#     ):
#         super().__init__(
#             model_max_length=model_max_length,
#             tokenizer=tokenizer,
#             use_clean_caption=use_clean_caption,
#             enable_text_preprocessing=enable_text_preprocessing,
#             padding_type=padding_type,
#             support_chinese=support_chinese,
#             cfg=cfg,
#         )

#         self.default_text_ratio = default_text_ratio

#     def drop(self, text):
#         rand_num = random.random()
#         rand_num_text = random.random()

#         if rand_num < self.cfg:
#             if rand_num_text < self.default_text_ratio:
#                 text = ["A scene with coherent and clear visuals." ]
#             else:
#                 text = [""]

#         return text
    
#     def __call__(self, texts):
#         if self.enable_text_preprocessing:
#             texts_info = [
#                 TextProcesser.text_preprocessing(text, self.use_clean_caption)
#                 for text in texts
#             ]
#             texts_info = self.drop(texts_info)
#         else:
#             texts_info = texts

#         text_tokens_and_mask = self.tokenizer(
#             texts_info,
#             max_length=self.model_max_length,
#             padding=self.padding,
#             truncation=True,
#             return_attention_mask=True,
#             add_special_tokens=True,
#             return_tensors="pt",
#         )
#         prompt_ids = text_tokens_and_mask["input_ids"]
#         prompt_mask = text_tokens_and_mask["attention_mask"]
#         return prompt_ids, prompt_mask

def filter_resolution(h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def get_seed_worker(seed):
    """Deterministic dataloader"""

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def collate_fn_default(batch):
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        input_ids = torch.cat(input_ids, dim=-1)
        use_mask = True
    elif "mask" in batch[0] and isinstance(batch[0]["mask"], torch.Tensor):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        masks = torch.cat(masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["input_ids"] = input_ids

    return ret


def build_iterations(train_dl=None, val_dl=None, test_dl=None, iterator_type="cyclic"):

    def _cyclic_iter(dl):
        while True:
            for x in dl:
                yield x
    
    def _get_iterator(dataloader, iter_type=iterator_type):
        """Return dataset iterator."""
        if iter_type == "single":
            return iter(dataloader)
        elif iter_type == "cyclic":
            return iter(_cyclic_iter(dataloader))
        else:
            raise NotImplementedError("unexpected iterator type")
    
    if train_dl is not None:
        train_data_iterator = _get_iterator(train_dl)
    else:
        train_data_iterator = None

    if val_dl is not None:
        valid_data_iterator = _get_iterator(val_dl)
    else:
        valid_data_iterator = None

    if test_dl is not None:
        test_data_iterator = _get_iterator(test_dl)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator

