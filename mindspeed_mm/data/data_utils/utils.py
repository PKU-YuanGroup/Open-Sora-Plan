# Modified from huggingface diffusers repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# TextProcesser: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py


import os
import re
import gc
import sys
import html
import math
import copy
import random
import urllib.parse as ul
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
from PIL import Image
from bs4 import BeautifulSoup
from einops import rearrange
from torchvision import get_video_backend
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
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

from mindspeed_mm.data.data_utils.data_transform import TemporalRandomCrop, Expand2Square
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.conversation import get_conv_template
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


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
        if data_path.endswith(".csv"):
            data_out = pd.read_csv(data_path)
            if return_type == "list":
                return data_out.to_dict("records")
            else:
                return data_out
        elif data_path.endswith(".json"):
            data_out = pd.read_json(data_path)
            return data_out.to_dict("records")
        elif data_path.endswith(".jsonl"):
            data_out = pd.read_json(data_path, lines=True)
            return data_out.to_dict("records")
        elif data_path.endswith(".parquat"):
            data_out = pd.read_parquat(data_path)
            return data_out.to_dict("records")
        else:
            raise NotImplementedError(f"Unsupported file format: {self.data_path}")

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


class DecordInit:
    """Using Decord (https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"
            f"num_threads={self.num_threads})"
        )
        return repr_str


class VideoReader:
    """support some methods to read video"""

    def __init__(self, video_reader_type=None, num_threads=1):
        self.video_reader_type = video_reader_type
        if self.video_reader_type == "decoder":
            self.v_decoder = DecordInit(num_threads)

    def __call__(self, video_path):
        is_decord_read = False
        info = None

        if self.video_reader_type == "decoder":
            vframes = self.v_decoder(video_path)
            is_decord_read = True
        elif self.video_reader_type == "torchvision":
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, pts_unit="sec", output_format="TCHW"
            )  # [T: temporal, C: channel, H: height, W: width]
        elif self.video_reader_type == "av":
            vframes, aframes, info = read_video_av(filename=video_path, pts_unit="sec", output_format="TCHW")
        else:
            raise NotImplementedError(
                f"Unsupported video reader type: {self.video_reader_type}"
            )
        return vframes, info, is_decord_read


def read_video_av(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = "pts", float / Fraction if pts_unit = "sec", optional):
            The start presentation time of the video
        end_pts (int if pts_unit = "pts", float / Fraction if pts_unit = "sec", optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either "pts" or "sec". Defaults to "pts".
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    if get_video_backend() != "pyav":
        vframes, aframes, info = _video_opt._read_video(filename, start_pts, end_pts, pts_unit)
    else:
        _check_av_available()

        if end_pts is None:
            end_pts = float("inf")

        if end_pts < start_pts:
            raise ValueError(
                f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
            )

        info = {}
        video_frames = []
        audio_frames = []
        audio_timebase = _video_opt.default_timebase

        container = av.open(filename, metadata_errors="ignore")
        try:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            if container.streams.video:
                video_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)

            if container.streams.audio:
                audio_frames = _read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.audio[0],
                    {"audio": 0},
                )
                info["audio_fps"] = container.streams.audio[0].rate
        except av.AVError as ex:
            raise ex
        finally:
            container.close()
            del container
            # NOTE: manually garbage collect to close pyav threads
            gc.collect()

        vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]
        aframes_list = [frame.to_ndarray() for frame in audio_frames]

        if vframes_list:
            vframes = torch.as_tensor(np.stack(vframes_list))
        else:
            vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

        if aframes_list:
            aframes = np.concatenate(aframes_list, 1)
            aframes = torch.as_tensor(aframes)
            if pts_unit == "sec":
                start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
                if end_pts != float("inf"):
                    end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
            aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
        else:
            aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


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
            drop_short_ratio=1.0,
            max_height=480,
            max_width=640,
            **kwargs,
    ):
        self.num_frames = num_frames
        self.train_pipeline = train_pipeline
        self.video_transforms = None
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
        self.data_storage_mode = data_storage_mode
        if self.data_storage_mode == "combine":
            self.train_fps = train_fps
            self.speed_factor = speed_factor
            self.drop_short_ratio = drop_short_ratio
            self.max_height = max_height
            self.max_width = max_width

    def __call__(self, vframes, num_frames=None, frame_interval=None, image_size=None, is_decord_read=False,
                 predefine_num_frames=13):
        if image_size:
            self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                                                   image_size=image_size)
        else:
            self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline)
        if self.data_storage_mode == "standard":
            total_frames = len(vframes)
            if num_frames:
                self.num_frames = num_frames
                self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            if end_frame_ind - start_frame_ind < self.num_frames:
                raise AssertionError("the video does not have enough frames.")
            frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
            )
            if is_decord_read:
                video = vframes.get_batch(frame_indice).asnumpy()
                video = torch.from_numpy(video)
                # THWC -> TCHW,  [T: temporal, C: channel, H: height, W: width]
                video = video.permute(0, 3, 1, 2)
            else:
                video = vframes[frame_indice]  # TCHW

            video = self.video_transforms(video)
            # TCHW -> CTHW
            video = video.permute(1, 0, 2, 3)
        else:
            video = self.combine_data_video_process(
                vframes,
                is_decord_read=is_decord_read,
                predefine_num_frames=predefine_num_frames,
            )
        return video

    def combine_data_video_process(
            self, vframes, is_decord_read=True, predefine_num_frames=13
    ):
        total_frames = len(vframes)
        fps = vframes.get_avg_fps() if vframes.get_avg_fps() > 0 else 30.0
        # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
        frame_interval = (
            1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        )
        # some special video should be set to a different number
        start_frame_idx = 0
        frame_indices = np.arange(start_frame_idx, total_frames, frame_interval).astype(
            int
        )
        frame_indices = frame_indices[frame_indices < total_frames]
        # speed up
        max_speed_factor = len(frame_indices) / self.num_frames
        if self.speed_factor > 1 and max_speed_factor > 1:
            speed_factor = min(self.speed_factor, max_speed_factor)
            target_frame_count = int(len(frame_indices) / speed_factor)
            speed_frame_idx = np.linspace(
                0, len(frame_indices) - 1, target_frame_count, dtype=int
            )
            frame_indices = frame_indices[speed_frame_idx]

        #  too long video will be temporal-crop randomly
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=4, model_ds_t=4
        )
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )
        frame_indices = frame_indices[:end_frame_idx]
        if predefine_num_frames != len(frame_indices):
            raise ValueError(
                f"predefine_num_frames ({predefine_num_frames}) is not equal with frame_indices ({len(frame_indices)})"
            )
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )
        video = vframes.get_batch(frame_indices).asnumpy()
        video = torch.from_numpy(video)
        # (T, H, W, C) -> (T C H W)
        video = video.permute(0, 3, 1, 2)

        h, w = video.shape[-2:]
        if h / w > 17 / 16 or h / w < 8 / 16:
            raise AssertionError(
                f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But the video found ratio is {round(h / w, 2)} with the shape of {video.shape}"
            )
        # TCHW -> TCHW
        video = self.video_transforms(video)
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return video

    def define_frame_index(self, cap_list):
        new_cap_list = []
        sample_num_frames = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_resolution_mismatch = 0
        cnt_movie = 0
        cnt_img = 0
        for i in cap_list:
            path = i["path"]
            cap = i.get("cap", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue
            if path.endswith(".mp4"):
                # ======no fps and duration=====
                duration = i.get("duration", None)
                fps = i.get("fps", None)
                if fps is None or duration is None:
                    continue

                # ======resolution mismatch=====
                resolution = i.get("resolution", None)
                if resolution is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    if (
                            resolution.get("height", None) is None
                            or resolution.get("width", None) is None
                    ):
                        cnt_no_resolution += 1
                        continue
                    height, width = i["resolution"]["height"], i["resolution"]["width"]
                    aspect = self.max_height / self.max_width
                    hw_aspect_thr = 1.5
                    max_h_div_w_ratio = hw_aspect_thr * aspect
                    min_h_div_w_ratio = 1 / hw_aspect_thr * aspect
                    is_pick = (
                            height / width <= max_h_div_w_ratio
                            and height / width >= min_h_div_w_ratio
                    )
                    if not is_pick:
                        cnt_resolution_mismatch += 1
                        continue

                i["num_frames"] = int(fps * duration)

                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = fps / self.train_fps
                start_frame_idx = (
                    8 if "/storage/dataset/movie" in i["path"] else 0
                )  # special video
                frame_indices = np.arange(
                    start_frame_idx, i["num_frames"], frame_interval
                ).astype(int)
                frame_indices = frame_indices[frame_indices < i["num_frames"]]

                # comment out it to enable dynamic frames training
                if (
                        len(frame_indices) < self.num_frames
                        and random.random() < self.drop_short_ratio
                ):
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                # to find a suitable end_frame_idx, to ensure we do not need pad video
                end_frame_idx = self.find_closest_y(
                    len(frame_indices), vae_stride_t=4, model_ds_t=4
                )
                if (
                        end_frame_idx == -1
                ):  # too short that can not be encoded exactly by videovae
                    cnt_too_short += 1
                    continue
                frame_indices = frame_indices[:end_frame_idx]

                if "/storage/dataset/movie" in i["path"]:
                    cnt_movie += 1
                i["sample_frame_index"] = frame_indices.tolist()
                new_cap_list.append(i)
                i["sample_num_frames"] = len(
                    i["sample_frame_index"]
                )  # will use in dataloader(group sampler)
                sample_num_frames.append(i["sample_num_frames"])
            elif path.endswith(".jpg"):  # image
                cnt_img += 1
                new_cap_list.append(i)
                i["sample_num_frames"] = 1
                sample_num_frames.append(i["sample_num_frames"])
            else:
                raise NameError(
                    f"Unknown file extention {path.split('.')[-1]}, only support .mp4 for video and .jpg for image"
                )

        print(
            f"no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short}, "
            f"no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, "
            f"Counter(sample_num_frames): {Counter(sample_num_frames)}, cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, "
            f"before filter: {len(cap_list)}, after filter: {len(new_cap_list)}"
        )
        return new_cap_list, sample_num_frames

    def find_closest_y(self, x, vae_stride_t=4, model_ds_t=4):
        for y in range(x, 12, -1):
            if (y - 1) % vae_stride_t == 0 and (
                    (y - 1) // vae_stride_t + 1
            ) % model_ds_t == 0:
                # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
                # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
                return y
        return -1


class ImageProcesser:
    """Used for image data preprocessing"""

    def __init__(
            self,
            num_frames=16,
            train_pipeline=None,
            image_reader_type="torchvision",
            image_processer_type="image2video",
            dynamic_image_size=False,
            image_size=224,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            use_thumbnail=False,
            **kwargs,
    ):
        self.num_frames = num_frames
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline
        )
        self.video_transforms = get_transforms(
            is_video=True, train_pipeline=train_pipeline
        )
        self.train_pipeline = train_pipeline
        self.image_reader_type = image_reader_type
        self.image_processer_type = image_processer_type
        self.dynamic_image_size = dynamic_image_size
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail

    def __call__(self, image_path, train_pipeline, mode, num_image):
        if self.image_processer_type == "image2video":
            image = self.image_to_video(image_path)
        elif self.image_processer_type == "image2image":
            image = self.image_to_image(image_path)
        elif self.image_processer_type == "image2pixel":
            image = self.image_to_pixel_values(image_path, train_pipeline, mode, num_image)
        else:
            raise NotImplementedError(
                f"Unsupported image processer type: {self.image_processer_type}"
            )
        return image

    def image_to_video(self, image_path):
        image = self.image_reader(image_path)
        image = self.image_transforms(image)
        video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        video = video.permute(1, 0, 2, 3)  # TCHW -> CTHW
        return video

    def image_to_image(self, image_path):
        image = self.image_reader(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  # [1 c h w]
        # [1 C H W] -> num_img [1 C H W]
        if "human_images" in image_path:
            image = self.image_transforms(image)
        else:
            image = self.video_transforms(image)
        # [1 C H W] -> [C 1 H W]
        image = image.permute(1, 0, 2, 3)
        return image

    def image_to_pixel_values(self, image_path, train_pipeline, mode="", num_image=1):
        image = self.image_reader(image_path)
        max_num = self.max_dynamic_patch // num_image if mode == "multi_image" else self.max_dynamic_patch
        if self.image_reader_type == "torchvision":
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=max_num,
                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            else:  # Otherwise, use the original image as a single patch
                images = [image]

            # Apply the transformation to each image and stack the results into a tensor
            pixel_values = [self.image_transforms(image) for image in images]
            pixel_values = pixel_values if mode == "multi_image" else torch.stack(pixel_values)
        else:
            if train_pipeline["pad2square"]:
                expand2square = Expand2Square(mean=train_pipeline["image_mean"])
                image = expand2square(image)
            processer = CLIPImageProcessor(**train_pipeline)
            pixel_values = processer.preprocess(image, return_tensors="pt", **train_pipeline)["pixel_values"][0]
        return pixel_values

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
            use_clean_caption=True,
            enable_text_preprocessing=True,
            padding_type="max_length",
            support_chinese=False,
            cfg=0.1,
    ):
        self.model_max_length = model_max_length
        self.padding = padding_type
        self.tokenizer = tokenizer
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
        return prompt_ids, prompt_mask

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


def get_seed_worker(seed):
    """Deterministic dataloader"""

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


class SingletonMeta(type):
    """
    This is a metaclass for creating singletons.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):
    """
    This is a data program for data multithreaded processing.
    """

    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        random.shuffle(self.elements)
        print(f"n_elements: {len(self.elements)}", flush=True)

        for i in range(self.num_workers):
            self.n_used_elements[i] = 0
            per_worker = int(math.ceil(len(self.elements) / float(self.num_workers)))
            start = i * per_worker
            end = min(start + per_worker, len(self.elements))
            self.worker_elements[i] = self.elements[start:end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][
            self.n_used_elements[worker_id] % len(self.worker_elements[worker_id])
            ]
        self.n_used_elements[worker_id] += 1
        return idx


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


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    This function finds the closest aspect ratio from a set of target aspect ratios based on the input
    image's aspect ratio. It calculates the difference between the input image's aspect ratio and each
    target aspect ratio, and returns the ratio that has the smallest difference. If two ratios have the same
    difference, it chooses the one whose area is closer to a specific size threshold.
    :param aspect_ratio: The aspect ratio of the input image, calculated as width / height.
    :param target_ratios: A list of target aspect ratios in the form of tuples, where each tuple represents a width-height ratio.
    :param width: The width of the input image.
    :param height: The height of the input image.
    :param image_size: A reference size used for comparing the areas of the input image and the target ratios.
    :return:best_ratio (tuple): The target aspect ratio that is closest to the input image's aspect ratio.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    This function dynamically preprocesses an input image by resizing it to match a closest target
    aspect ratio and then splitting the resized image into smaller blocks. It optionally generates
    a thumbnail version of the image. The preprocessing is useful for adjusting the input data for tasks like
    data augmentation or image classification in machine learning.
    :param image:The input image to be processed.
    :param min_num: The minimum number of blocks used to create target aspect ratios.
    :param max_num:The maximum number of blocks used to create target aspect ratios.
    :param image_size:The size to which the image should be resized before splitting into blocks.
    :param use_thumbnail: If True, a thumbnail version of the image will be generated and added to the list of
                          processed images when the number of blocks is greater than 1.
    :return:processed_images (list of PIL.Image): A list of processed images, including blocks of the resized image,
                    and optionally a thumbnail image. The number of blocks is determined by the target aspect ratio.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def preprocess_multimodal(
        sources: Sequence[str],
        is_multimodal,
        mm_use_im_start_end,
) -> Dict:
    """
    Process multimodal sources by handling image tokens.
    """
    image_token = MODEL_CONSTANTS['llava']["IMAGE_TOKEN"]
    img_start_token = MODEL_CONSTANTS['llava']["IMG_START_TOKEN"]
    img_end_token = MODEL_CONSTANTS['llava']["IMG_END_TOKEN"]

    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if image_token in sentence["value"]:
                sentence["value"] = sentence["value"].replace(image_token, "").strip()
                sentence["value"] = image_token + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            replace_token = image_token
            if mm_use_im_start_end:
                replace_token = img_start_token + replace_token + img_end_token
            sentence["value"] = sentence["value"].replace(image_token, replace_token)

    return sources


def preprocess_v1(
        sources,
        is_multimodal,
        mm_use_im_start_end,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = True
) -> Dict:
    """
    Process sources for llava-v1 of the preprocessing pipeline.
    """
    sources = preprocess_multimodal(sources, is_multimodal, mm_use_im_start_end)

    ignore_index = MODEL_CONSTANTS['llava']["IGNORE_INDEX"]
    image_token_index = MODEL_CONSTANTS['llava']["IMAGE_TOKEN_INDEX"]
    conv = get_conv_template("llava-v1")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = get_formatted_conversations(sources, roles, conv)

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
             for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = ignore_index
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer, image_token_index))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer, image_token_index)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = ignore_index

            cur_len += round_len
        target[cur_len:] = ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = ignore_index
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
    )


def preprocess_plain(
        sources: Sequence[str],
        is_multimodal,
        mm_use_im_start_end,
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """
    Process plain text sources for preprocessing.
    """
    sources = preprocess_multimodal(sources, is_multimodal, mm_use_im_start_end)

    image_token_index = MODEL_CONSTANTS['llava']["IMAGE_TOKEN_INDEX"]
    image_token = MODEL_CONSTANTS['llava']["IMAGE_TOKEN"]
    ignore_index = MODEL_CONSTANTS['llava']["IGNORE_INDEX"]
    conv = get_conv_template("llava-plain")
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        source[0]["value"] = image_token
        conversation = source[0]["value"] + source[1]["value"] + conv.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
                 for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer, image_token_index=image_token_index))
        target[:tokenized_len] = ignore_index

    return dict(input_ids=input_ids[0], labels=targets[0])


def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        num_image: int = 1
) -> Dict:
    """
    Process sources for internvl model preprocessing.
    """
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = get_formatted_conversations(sources, roles, conv)

    im_start_token = MODEL_CONSTANTS['internvl']["IMG_START_TOKEN"]
    im_context_token = MODEL_CONSTANTS['internvl']["IMG_CONTEXT_TOKEN"]
    im_end_token = MODEL_CONSTANTS['internvl']["IMG_END_TOKEN"]

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{im_start_token}{im_context_token * num_image_token_list[i]}{im_end_token}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.", flush=True)

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
        attention_mask=input_ids.ne(tokenizer.pad_token_id)[0],
    )


def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    """
    Tokenize prompts with image tokens.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_formatted_conversations(sources, roles, conv):
    """
    Format conversations based on provided roles and conversation template.
    """
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                raise ValueError(
                    f"Role mismatch at {sentence}, expected {conv.roles[j % 2]}, got {role}")
            sentence["value"] = sentence["value"].strip()
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    return conversations


def preprocess(
        template_name,
        sources,
        tokenizer,
        num_image_token_list,
        group_by_length,
        is_multimodal,
        mm_use_im_start_end
):
    """
    Select and run the appropriate preprocessing function based on template name.
    """
    if template_name == "internlm2-chat":
        ret = preprocess_internlm(template_name, sources,
                                  tokenizer, num_image_token_list,
                                  group_by_length=group_by_length)
    elif template_name == "llava-v1":
        ret = preprocess_v1(
            sources,
            is_multimodal,
            mm_use_im_start_end,
            tokenizer,
            has_image=True)
    elif template_name == "llava-plain":
        ret = preprocess_plain(
            sources,
            is_multimodal,
            mm_use_im_start_end,
            tokenizer)
    else:
        raise ValueError("%s preprocessor is not implemented" % type(template_name))
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

