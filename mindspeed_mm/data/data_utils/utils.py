# Modified from huggingface diffusers repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# TextProcesser: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py


import html
import math
import os
import random
import re
import string
import urllib.parse as ul

try:
    import decord
except ImportError:
    print("Failed to import decord module.")
from collections import Counter

import ftfy
import numpy as np
import pandas as pd
import torch
import torchvision
from bs4 import BeautifulSoup
from einops import rearrange
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from mindspeed_mm.data.data_utils.data_transform import TemporalRandomCrop
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class DataFileReader:
    """get the data from different types of files such as csv/json/parquat"""

    def __init__(self, data_storage_mode="standard"):
        self.data_storage_mode = data_storage_mode

    def __call__(self, data_path):
        if self.data_storage_mode == "standard":
            return self.get_datasamples()
        elif self.data_storage_mode == "combine":
            return self.get_cap_list(data_path)
        else:
            raise NotImplementedError("Not support now.")

    def get_datasamples(self, data_path):
        if data_path.endswith(".csv"):
            data_out = pd.read_csv(data_path)
            return data_out.to_dict("records")
        elif data_path.endswith(".json"):
            data_out = pd.read_json(data_path)
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
        if self.video_reader_type == "decoder":
            vframes = self.v_decoder(video_path)
            is_decord_read = True
        elif self.video_reader_type == "torchvision":
            vframes, aframes, info = torchvision.io.read_video(
                filename=video_path, pts_unit="sec", output_format="TCHW"
            )  # [T: temporal, C: channel, H: height, W: width]
            is_decord_read = False
        else:
            raise NotImplementedError(
                f"Unsupported video reader type: {self.video_reader_type}"
            )
        return vframes, is_decord_read


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
        self.video_transforms = get_transforms(
            is_video=True, train_pipeline=train_pipeline
        )
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)
        self.data_storage_mode = data_storage_mode
        if self.data_storage_mode == "combine":
            self.train_fps = train_fps
            self.speed_factor = speed_factor
            self.drop_short_ratio = drop_short_ratio
            self.max_height = max_height
            self.max_width = max_width

    def __call__(self, vframes, is_decord_read=False, predefine_num_frames=13):
        if self.data_storage_mode == "standard":
            total_frames = len(vframes)
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

    # TODO
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
        **kwargs,
    ):
        self.num_frames = num_frames
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline
        )
        self.video_transforms = get_transforms(
            is_video=True, train_pipeline=train_pipeline
        )
        self.image_reader_type = image_reader_type
        self.image_processer_type = image_processer_type

    def __call__(self, image_path):
        if self.image_processer_type == "image2video":
            image = self.image_to_video(image_path)
        elif self.image_processer_type == "image2image":
            image = self.image_to_image(image_path)
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

    def image_reader(self, image_path):
        if self.image_reader_type == "torchvision":
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

    def __call__(self, texts):
        texts_info = [
            TextProcesser.text_preprocessing(text, self.use_clean_caption)
            for text in texts
        ]
        texts_info = texts_info if random.random() > self.cfg else [""]
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


# TODO
def get_batch_on_this_tp_rank(data_iterator):
    """
    :param data_iterator:
    :return:
    """

    batch = data_iterator
    return batch
