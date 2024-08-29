import argparse
import ast
import asyncio
import functools
import gc
import glob
import hashlib
import importlib
import json
import math
import os
import pathlib
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from textwrap import dedent, indent
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import safetensors.torch
import toml
import torch
import torch_npu
import transformers
import voluptuous
from accelerate import Accelerator, InitProcessGroupKwargs
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from voluptuous import Any, ExactSequence, MultipleInvalid, Object, Required, Schema

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".WEBP",
    ".BMP",
]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except ImportError:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except ImportError:
    pass

# JPEG-XL on Windows
try:
    import pillow_jxl

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except ImportError:
    pass


IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"


def load_image(image_path):

    with Image.open(image_path) as image:
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image, np.uint8)
        return img


# Loads an image. Return value numpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: Image.Image, reso, resized_size: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        # Resize
        image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[p : p + reso[1]]

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    if image.shape[0] != reso[1] and image.shape[1] != reso[0]:
        raise ValueError(f"internal error, illegal trimmed size: {image.shape}, {reso}")
    return image, original_size, crop_ltrb


class ImageInfo:
    def __init__(
        self,
        image_key: str,
        num_repeats: int,
        caption: str,
        is_reg: bool,
        absolute_path: str,
    ) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: torch.Tensor = None
        self.latents_flipped: torch.Tensor = None
        self.latents_npz: str = None
        self.latents_original_size: Tuple[int, int] = (
            None  # original image size, not latents size
        )
        self.latents_crop_ltrb: Tuple[int, int] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: str = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        # SDXL, optional
        self.text_encoder_outputs_npz: Optional[str] = None
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                if max_size < max_reso[0]:
                    raise ValueError(
                        "the max_size should be larger than the width of max_reso"
                    )
                if max_size < max_reso[1]:
                    raise ValueError(
                        "the max_size should be larger than the height of max_reso"
                    )
            if min_size is not None:
                if max_size < min_size:
                    raise ValueError("the max_size should be larger than the min_size")

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = (
            []
        )  # On preprocessing (image key, image, original size, crop left / top) Key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id.get(reso, None)
        if bucket_id is not None:
            self.buckets[bucket_id].append(image_or_info)
        else:
            pass

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id.get(reso, None)
            if bucket_id is not None:
                sorted_buckets.append(self.buckets[bucket_id])
                sorted_reso_to_id[reso] = i
            else:
                pass

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = make_bucket_resolutions(
            self.max_reso, self.min_size, self.max_size, self.reso_steps
        )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # print(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(
                    ar_errors
                ).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (
                int(image_width * scale + 0.5),
                int(image_height * scale + 0.5),
            )
            # print("use predef", image_width, image_height, reso, resized_size)
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                if abs(resized_width / resized_height - aspect_ratio) >= 1e-2:
                    raise ValueError("aspect is illegal")

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # print(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # print(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(
                    ar_height_rounded - aspect_ratio
                ):
                    resized_size = (
                        b_width_rounded,
                        int(b_width_rounded / aspect_ratio + 0.5),
                    )
                else:
                    resized_size = (
                        int(b_height_rounded * aspect_ratio + 0.5),
                        b_height_rounded,
                    )
                # print(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # print("use arbitrary", image_width, image_height, resized_size, bucket_width, bucket_height)

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(
        self, use_color_aug: bool
    ):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        num_repeats: int,
        shuffle_caption: bool,
        caption_separator: str,
        keep_tokens: int,
        keep_tokens_separator: str,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        caption_prefix: Optional[str],
        caption_suffix: Optional[str],
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
    ) -> None:
        self.image_dir = image_dir
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる

        self.img_count = 0


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        num_repeats,
        shuffle_caption,
        caption_separator: str,
        keep_tokens,
        keep_tokens_separator,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        if image_dir is None:
            raise ValueError("image_dir must be specified")

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer]],
        max_token_length: int,
        resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        debug_dataset: bool,
    ) -> None:
        super().__init__()

        self.tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        self.max_token_length = max_token_length
        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.tokenizer_max_length = (
            self.tokenizers[0].model_max_length
            if max_token_length is None
            else max_token_length + 2
        )

        self.current_epoch: int = (
            0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ
        )

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if (
            not self.current_epoch == epoch
        ):  # epochが切り替わったらバケツをシャッフルする
            self.shuffle_buckets()
        self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # caption に prefix/suffix を付ける
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = (
            subset.caption_dropout_rate > 0
            and random.random() < subset.caption_dropout_rate
        )
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            if (
                subset.shuffle_caption
                or subset.token_warmup_step > 0
                or subset.caption_tag_dropout_rate > 0
            ):
                fixed_tokens = []
                flex_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(
                        subset.keep_tokens_separator, 1
                    )
                    fixed_tokens = [
                        t.strip()
                        for t in fixed_part.split(subset.caption_separator)
                        if t.strip()
                    ]
                    flex_tokens = [
                        t.strip()
                        for t in flex_part.split(subset.caption_separator)
                        if t.strip()
                    ]
                else:
                    tokens = [
                        t.strip()
                        for t in caption.strip().split(subset.caption_separator)
                    ]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(
                        subset.token_warmup_step * self.max_train_steps
                    )
                if (
                    subset.token_warmup_step
                    and self.current_step < subset.token_warmup_step
                ):
                    tokens_len = (
                        math.floor(
                            (self.current_step)
                            * (
                                (len(flex_tokens) - subset.token_warmup_min)
                                / (subset.token_warmup_step)
                            )
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    tags = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            tags.append(token)
                    return tags

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if isinstance(str_to, list):
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).input_ids

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1,
                    self.tokenizer_max_length - tokenizer.model_max_length + 2,
                    tokenizer.model_max_length - 2,
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(
                    1,
                    self.tokenizer_max_length - tokenizer.model_max_length + 2,
                    tokenizer.model_max_length - 2,
                ):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if (
                        ids_chunk[-2] != tokenizer.eos_token_id
                        and ids_chunk[-2] != tokenizer.pad_token_id
                    ):
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        print("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        if self.enable_bucket:
            print("make buckets")
        else:
            print("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        if self.enable_bucket:
            if (
                self.bucket_manager is None
            ):  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    print(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = (
                    self.bucket_manager.select_bucket(image_width, image_height)
                )

                # print(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(
                False, (self.width, self.height), None, None, None
            )
            self.bucket_manager.set_predefined_resos(
                [(self.width, self.height)]
            )  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = (
                    self.bucket_manager.select_bucket(image_width, image_height)
                )

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(
                    image_info.bucket_reso, image_info.image_key
                )

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            print(
                "number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）"
            )
            for i, (reso, bucket) in enumerate(
                zip(self.bucket_manager.resos, self.bucket_manager.buckets)
            ):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {
                        "resolution": reso,
                        "count": len(bucket),
                    }
                    print(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            print(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List(BucketBatchIndex) = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(
                    BucketBatchIndex(bucket_index, self.batch_size, batch_index)
                )

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        if (
            self.bucket_reso_steps is not None
            and self.bucket_reso_steps % min_steps != 0
        ):
            raise ValueError(
                f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
                + f"bucket_reso_stepsが{self.bucket_reso_steps}です。{min_steps}で割り切れる必要があります"
            )

    def is_latent_cacheable(self):
        return all(
            [not subset.color_aug and not subset.random_crop for subset in self.subsets]
        )

    def is_text_encoder_output_cacheable(self):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def cache_latents(
        self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True
    ):
        # マルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        print("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution
        batches = []
        batch = []
        print("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(
                    info.bucket_reso, info.latents_npz, subset.flip_aug
                )

                if cache_available:  # do not add to batch
                    continue

            # if last member of batch has different resolution, flush the batch
            if len(batch) > 0 and batch[-1].bucket_reso != info.bucket_reso:
                batches.append(batch)
                batch = []

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if (
            cache_to_disk and not is_main_process
        ):  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        print("caching latents...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            cache_batch_latents(
                vae, cache_to_disk, batch, subset.flip_aug, subset.random_crop
            )

    # weight_dtype
    def cache_text_encoder_outputs(
        self,
        tokenizers,
        text_encoders,
        device,
        weight_dtype,
        cache_to_disk=False,
        is_main_process=True,
    ):
        if len(tokenizers) != 2:
            raise ValueError("only support SDXL")

        print("caching text encoder outputs.")
        image_infos = list(self.image_data.values())

        print("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            # subset = self.image_to_subset[info.image_key]
            if cache_to_disk:
                te_out_npz = (
                    os.path.splitext(info.absolute_path)[0]
                    + TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX
                )
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:  # store to info only
                    continue

                if os.path.exists(te_out_npz):
                    continue

            image_infos_to_cache.append(info)

        if (
            cache_to_disk and not is_main_process
        ):  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # prepare tokenizers and text encoders
        for text_encoder in text_encoders:
            text_encoder.to(device)
            if weight_dtype is not None:
                text_encoder.to(dtype=weight_dtype)

        # create batch
        batch = []
        batches = []
        for info in image_infos_to_cache:
            input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
            input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
            batch.append((info, input_ids1, input_ids2))

            if len(batch) >= self.batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        print("caching text encoder outputs...")
        for batch in tqdm(batches):
            infos, input_ids1, input_ids2 = zip(*batch)
            input_ids1 = torch.stack(input_ids1, dim=0)
            input_ids2 = torch.stack(input_ids2, dim=0)
            cache_batch_text_encoder_outputs(
                infos,
                tokenizers,
                text_encoders,
                self.max_token_length,
                cache_to_disk,
                input_ids1,
                input_ids2,
                weight_dtype,
            )

    def get_image_size(self, image_path):

        with Image.open(image_path) as image:
            return image.size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str):
        img = load_image(image_path)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(
            self.height / height, self.width / width
        )  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(
            1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1]))
        )  # 指定した顔最小サイズ
        max_scale = min(
            1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0]))
        )  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        if nh < self.height and nw < self.width:
            raise ValueError(f"internal error. small scale {scale}, {width}*{height}")

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(
            zip((self.height, self.width), (height, width), (face_cy, face_cx))
        ):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                im_range = max(
                    length - face_p, face_p
                )  # 画像の端から顔中心までの距離の長いほう
                p1 = (
                    p1
                    + (random.randint(0, im_range) + random.randint(0, im_range))
                    - im_range
                )  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if (
            self.caching_mode is not None
        ):  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        input_ids2_list = []
        latents_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs1_list = []
        text_encoder_outputs2_list = []
        text_encoder_pool2_list = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data.get(image_key, None)
            subset = self.image_to_subset.get(image_key, None)
            loss_weights.append(
                self.prior_loss_weight if image_info.is_reg else 1.0
            )  # in case of fine tuning, is_reg is always False

            flipped = (
                subset.flip_aug and random.random() < 0.5
            )  # not flipped or flipped with 50% chance

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                else:
                    latents = image_info.latents_flipped

                image = None
            elif (
                image_info.latents_npz is not None
            ):  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                latents, original_size, crop_ltrb, flipped_latents = (
                    load_latents_from_disk(image_info.latents_npz)
                )
                if flipped:
                    latents = flipped_latents
                    del flipped_latents
                latents = torch.FloatTensor(latents)

                image = None
            else:
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    subset, image_info.absolute_path
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                    )
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(
                            subset, img, face_cx, face_cy, face_w, face_h
                        )
                    elif im_h > self.height or im_w > self.width:
                        if not subset.random_crop:
                            raise ValueError(
                                f"image too large, but cropping and bucketing are disabled : {image_info.absolute_path}"
                            )
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    if im_h != self.height and im_w != self.width:
                        raise ValueError(
                            f"image size is small: {image_info.absolute_path}"
                        )

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    try:
                        img = aug(image=img)["image"]
                    except KeyError:
                        print("Warning: Augmentation didn't return an 'image' key")

                if flipped:
                    img = img[
                        :, ::-1, :
                    ].copy()  # copy to avoid negative stride problem

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる

            images.append(image)
            latents_list.append(latents)

            target_size = (
                (image.shape[2], image.shape[1])
                if image is not None
                else (latents.shape[2] * 8, latents.shape[1] * 8)
            )

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # captionとtext encoder outputを処理する
            caption = image_info.caption  # default
            if image_info.text_encoder_outputs1 is not None:
                text_encoder_outputs1_list.append(image_info.text_encoder_outputs1)
                text_encoder_outputs2_list.append(image_info.text_encoder_outputs2)
                text_encoder_pool2_list.append(image_info.text_encoder_pool2)
                captions.append(caption)
            elif image_info.text_encoder_outputs_npz is not None:
                text_encoder_outputs1, text_encoder_outputs2, text_encoder_pool2 = (
                    load_text_encoder_outputs_from_disk(
                        image_info.text_encoder_outputs_npz
                    )
                )
                text_encoder_outputs1_list.append(text_encoder_outputs1)
                text_encoder_outputs2_list.append(text_encoder_outputs2)
                text_encoder_pool2_list.append(text_encoder_pool2)
                captions.append(caption)
            else:
                caption = self.process_caption(subset, image_info.caption)
                if self.XTI_layers:
                    caption_layer = []
                    for layer in self.XTI_layers:
                        token_strings_from = " ".join(self.token_strings)
                        token_strings_to = " ".join(
                            [f"{x}_{layer}" for x in self.token_strings]
                        )
                        caption_ = caption.replace(token_strings_from, token_strings_to)
                        caption_layer.append(caption_)
                    captions.append(caption_layer)
                else:
                    captions.append(caption)

                if (
                    not self.token_padding_disabled
                ):  # this option might be omitted in future
                    if self.XTI_layers:
                        token_caption = self.get_input_ids(
                            caption_layer, self.tokenizers[0]
                        )
                    else:
                        token_caption = self.get_input_ids(caption, self.tokenizers[0])
                    input_ids_list.append(token_caption)

                    if len(self.tokenizers) > 1:
                        if self.XTI_layers:
                            token_caption2 = self.get_input_ids(
                                caption_layer, self.tokenizers[1]
                            )
                        else:
                            token_caption2 = self.get_input_ids(
                                caption, self.tokenizers[1]
                            )
                        input_ids2_list.append(token_caption2)

        example = {}
        example["loss_weights"] = torch.FloatTensor(loss_weights)

        if len(text_encoder_outputs1_list) == 0:
            if self.token_padding_disabled:
                # padding=True means pad in the batch
                example["input_ids"] = self.tokenizer[0](
                    captions, padding=True, truncation=True, return_tensors="pt"
                ).input_ids
                if len(self.tokenizers) > 1:
                    example["input_ids2"] = self.tokenizer[1](
                        captions, padding=True, truncation=True, return_tensors="pt"
                    ).input_ids
                else:
                    example["input_ids2"] = None
            else:
                example["input_ids"] = torch.stack(input_ids_list)
                example["input_ids2"] = (
                    torch.stack(input_ids2_list) if len(self.tokenizers) > 1 else None
                )
            example["text_encoder_outputs1_list"] = None
            example["text_encoder_outputs2_list"] = None
            example["text_encoder_pool2_list"] = None
        else:
            example["input_ids"] = None
            example["input_ids2"] = None

            example["text_encoder_outputs1_list"] = torch.stack(
                text_encoder_outputs1_list
            )
            example["text_encoder_outputs2_list"] = torch.stack(
                text_encoder_outputs2_list
            )
            example["text_encoder_pool2_list"] = torch.stack(text_encoder_pool2_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = (
            torch.stack(latents_list) if latents_list[0] is not None else None
        )
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack(
            [torch.LongTensor(x) for x in original_sizes_hw]
        )
        example["crop_top_lefts"] = torch.stack(
            [torch.LongTensor(x) for x in crop_top_lefts]
        )
        example["target_sizes_hw"] = torch.stack(
            [torch.LongTensor(x) for x in target_sizes_hw]
        )
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor(
            [self.network_multiplier] * len(captions)
        )
        example["original_sizes"] = original_sizes_hw
        example["crop_top_lefts_list"] = crop_top_lefts
        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data.get(image_key, None)
            subset = self.image_to_subset.get(image_key, None)

            if flip_aug is None:
                flip_aug = subset.flip_aug
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                if flip_aug != subset.flip_aug:
                    raise ValueError("flip_aug must be same in a batch")
                if random_crop != subset.random_crop:
                    raise ValueError("random_crop must be same in a batch")
                if bucket_reso != image_info.bucket_reso:
                    raise ValueError("bucket_reso must be same in a batch")

            caption = (
                image_info.caption
            )  # needs cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
    ) -> None:
        super().__init__(
            tokenizer, max_token_length, resolution, network_multiplier, debug_dataset
        )

        if resolution is None:
            raise ValueError("resolution is required")

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            if min(resolution) < min_bucket_reso:
                raise ValueError(
                    "min_bucket_reso must be equal or less than resolution"
                )
            if max(resolution) > max_bucket_reso:
                raise ValueError(
                    "max_bucket_reso must be equal or greater than resolution"
                )

            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension):
            # captionの候補ファイル名を作る
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [
                base_name + caption_extension,
                base_name_face_det + caption_extension,
            ]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            print(
                                f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}"
                            )
                            raise e
                        if len(lines) <= 0:
                            raise ValueError(f"caption file is empty: {cap_path}")
                        caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                print(f"not directory: {subset.image_dir}")
                return [], []

            img_paths = glob_images(subset.image_dir, "*")
            print(
                f"found directory {subset.image_dir} contains {len(img_paths)} image files"
            )

            # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
            captions = []
            missing_captions = []
            for img_path in img_paths:
                cap_for_img = read_caption(img_path, subset.caption_extension)
                if cap_for_img is None and subset.class_tokens is None:
                    print(
                        f"neither caption file nor class tokens are found. use empty caption for {img_path} / キャプションファイルもclass tokenも見つかりませんでした。空のキャプションを使用します: {img_path}"
                    )
                    captions.append("")
                    missing_captions.append(img_path)
                else:
                    if cap_for_img is None:
                        captions.append(subset.class_tokens)
                        missing_captions.append(img_path)
                    else:
                        captions.append(cap_for_img)

            self.set_tag_frequency(
                os.path.basename(subset.image_dir), captions
            )  # タグ頻度を記録

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = (
                    number_of_missing_captions - number_of_missing_captions_to_show
                )

                print(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}枚の画像にキャプションファイルが見つかりませんでした。これらの画像についてはキャプションなしで学習を続行します。class tokenが存在する場合はそれを使います。"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        print(
                            missing_caption
                            + f"... and {remaining_missing_captions} more"
                        )
                        break
                    print(missing_caption)
            return img_paths, captions

        print("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[ImageInfo] = []
        for subset in subsets:
            if subset.num_repeats < 1:
                print(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            img_paths, captions = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                print(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / 画像が見つからないためサブセットを無視します"
                )
                continue

            if subset.is_reg:
                num_reg_images += subset.num_repeats * len(img_paths)
            else:
                num_train_images += subset.num_repeats * len(img_paths)

            for img_path, caption in zip(img_paths, captions):
                info = ImageInfo(
                    img_path, subset.num_repeats, caption, subset.is_reg, img_path
                )
                if subset.is_reg:
                    reg_infos.append(info)
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        print(f"{num_train_images} train images with repeating.")
        self.num_train_images = num_train_images

        print(f"{num_reg_images} reg images.")
        if num_train_images < num_reg_images:
            print(
                "some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります"
            )

        if num_reg_images == 0:
            print("no regularization images / 正則化画像が見つかりませんでした")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < num_train_images:
                for info in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[DreamBoothDataset]):
        self.datasets: List[DreamBoothDataset]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # needs handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(
        self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            print(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs(
        self,
        tokenizers,
        text_encoders,
        device,
        weight_dtype,
        cache_to_disk=False,
        is_main_process=True,
    ):
        for i, dataset in enumerate(self.datasets):
            print(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(
                tokenizers,
                text_encoders,
                device,
                weight_dtype,
                cache_to_disk,
                is_main_process,
            )

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all(
            [dataset.is_text_encoder_output_cacheable() for dataset in self.datasets]
        )

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


# collate_fn use epoch,stepはmultiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = (
            dataset  # not used if worker_info is not None, in case of multiprocessing
        )

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]


def add_config_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_config",
        type=Path,
        default=None,
        help="config file for detail settings / 詳細な設定用の設定ファイル",
    )


# needs inherit Params class in Subset, Dataset


@dataclass
class BaseSubsetParams:
    image_dir: Optional[str] = None
    num_repeats: int = 1
    shuffle_caption: bool = False
    caption_separator: tuple = (",",)
    keep_tokens: int = 0
    keep_tokens_separator: tuple = (None,)
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range: Optional[Tuple[float, float]] = None
    random_crop: bool = False
    caption_prefix: Optional[str] = None
    caption_suffix: Optional[str] = None
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    token_warmup_min: int = 1
    token_warmup_step: int = 0


@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
    is_reg: bool = False
    class_tokens: Optional[str] = None
    caption_extension: str = ".txt"


@dataclass
class FineTuningSubsetParams(BaseSubsetParams):
    metadata_file: Optional[str] = None


@dataclass
class ControlNetSubsetParams(BaseSubsetParams):
    conditioning_data_dir: str = None
    caption_extension: str = ".caption"


@dataclass
class BaseDatasetParams:
    tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer]] = None
    max_token_length: int = None
    resolution: Optional[Tuple[int, int]] = None
    network_multiplier: float = 1.0
    debug_dataset: bool = False


@dataclass
class DreamBoothDatasetParams(BaseDatasetParams):
    batch_size: int = 1
    enable_bucket: bool = False
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    bucket_reso_steps: int = 64
    bucket_no_upscale: bool = False
    prior_loss_weight: float = 1.0


@dataclass
class BaseSubsetParams:
    image_dir: Optional[str] = None
    num_repeats: int = 1
    shuffle_caption: bool = False
    caption_separator: tuple = (",",)
    keep_tokens: int = 0
    keep_tokens_separator: tuple = (None,)
    color_aug: bool = False
    flip_aug: bool = False
    face_crop_aug_range: Optional[Tuple[float, float]] = None
    random_crop: bool = False
    caption_prefix: Optional[str] = None
    caption_suffix: Optional[str] = None
    caption_dropout_rate: float = 0.0
    caption_dropout_every_n_epochs: int = 0
    caption_tag_dropout_rate: float = 0.0
    token_warmup_min: int = 1
    token_warmup_step: int = 0


@dataclass
class SubsetBlueprint:
    params: DreamBoothSubsetParams


@dataclass
class DatasetBlueprint:
    is_dreambooth: bool
    is_controlnet: bool
    params: DreamBoothDatasetParams
    subsets: Sequence[SubsetBlueprint]


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    dataset_group: DatasetGroupBlueprint


class ConfigSanitizer:

    @staticmethod
    def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    @staticmethod
    def __validate_and_convert_scalar_or_twodim(
        klass, value: Union[float, Sequence]
    ) -> Tuple:
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            Schema(klass)(value)
            return (value, value)
        except Exception as e:
            if not isinstance(e, KeyboardInterrupt):
                return ConfigSanitizer.__validate_and_convert_twodim(klass, value)
            else:
                raise e

    # subset schema
    SUBSET_ASCENDABLE_SCHEMA = {
        "color_aug": bool,
        "face_crop_aug_range": functools.partial(
            __validate_and_convert_twodim.__func__, float
        ),
        "flip_aug": bool,
        "num_repeats": int,
        "random_crop": bool,
        "shuffle_caption": bool,
        "keep_tokens": int,
        "keep_tokens_separator": str,
        "token_warmup_min": int,
        "token_warmup_step": Any(float, int),
        "caption_prefix": str,
        "caption_suffix": str,
    }
    # DO means DropOut
    DO_SUBSET_ASCENDABLE_SCHEMA = {
        "caption_dropout_every_n_epochs": int,
        "caption_dropout_rate": Any(float, int),
        "caption_tag_dropout_rate": Any(float, int),
    }
    # DB means DreamBooth
    DB_SUBSET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "class_tokens": str,
    }
    DB_SUBSET_DISTINCT_SCHEMA = {
        Required("image_dir"): str,
        "is_reg": bool,
    }
    # FT means FineTuning
    FT_SUBSET_DISTINCT_SCHEMA = {
        Required("metadata_file"): str,
        "image_dir": str,
    }
    CN_SUBSET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
    }
    CN_SUBSET_DISTINCT_SCHEMA = {
        Required("image_dir"): str,
        Required("conditioning_data_dir"): str,
    }

    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "batch_size": int,
        "bucket_no_upscale": bool,
        "bucket_reso_steps": int,
        "enable_bucket": bool,
        "max_bucket_reso": int,
        "min_bucket_reso": int,
        "resolution": functools.partial(
            __validate_and_convert_scalar_or_twodim.__func__, int
        ),
        "network_multiplier": float,
    }

    # options handled by argparse but not handled by user config
    ARGPARSE_SPECIFIC_SCHEMA = {
        "debug_dataset": bool,
        "max_token_length": Any(None, int),
        "prior_loss_weight": Any(float, int),
    }
    # for handling default None value of argparse
    ARGPARSE_NULLABLE_OPTNAMES = [
        "face_crop_aug_range",
        "resolution",
    ]
    # prepare map because option name may differ among argparse and user config
    ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME = {
        "train_batch_size": "batch_size",
        "dataset_repeats": "num_repeats",
    }

    def __init__(
        self,
        support_dreambooth: bool,
        support_finetuning: bool,
        support_controlnet: bool,
        support_dropout: bool,
    ) -> None:
        if not (support_dreambooth or support_finetuning or support_controlnet):
            raise ValueError(
                "Neither DreamBooth mode nor fine tuning mode specified. Please specify one mode or more."
            )

        self.db_subset_schema = self.__merge_dict(
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DB_SUBSET_DISTINCT_SCHEMA,
            self.DB_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
        )

        self.ft_subset_schema = self.__merge_dict(
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.FT_SUBSET_DISTINCT_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
        )

        self.cn_subset_schema = self.__merge_dict(
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.CN_SUBSET_DISTINCT_SCHEMA,
            self.CN_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
        )

        self.db_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DB_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
            {"subsets": [self.db_subset_schema]},
        )

        self.ft_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
            {"subsets": [self.ft_subset_schema]},
        )

        self.cn_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.CN_SUBSET_ASCENDABLE_SCHEMA,
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
            {"subsets": [self.cn_subset_schema]},
        )

        if support_dreambooth and support_finetuning:

            def validate_flex_dataset(dataset_config: dict):
                subsets_config = dataset_config.get("subsets", [])

                if support_controlnet and all(
                    ["conditioning_data_dir" in subset for subset in subsets_config]
                ):
                    return Schema(self.cn_dataset_schema)(dataset_config)
                # check dataset meets FT style
                # NOTE: all FT subsets should have "metadata_file"
                elif all(["metadata_file" in subset for subset in subsets_config]):
                    return Schema(self.ft_dataset_schema)(dataset_config)
                # check dataset meets DB style
                # NOTE: all DB subsets should have no "metadata_file"
                elif all(["metadata_file" not in subset for subset in subsets_config]):
                    return Schema(self.db_dataset_schema)(dataset_config)
                else:
                    raise voluptuous.Invalid(
                        "DreamBooth subset and fine tuning subset cannot be mixed in the same dataset. Please split them into separate datasets. / DreamBoothのサブセットとfine tuninのサブセットを同一のデータセットに混在させることはできません。別々のデータセットに分割してください。"
                    )

            self.dataset_schema = validate_flex_dataset
        elif support_dreambooth:
            self.dataset_schema = self.db_dataset_schema
        elif support_finetuning:
            self.dataset_schema = self.ft_dataset_schema
        elif support_controlnet:
            self.dataset_schema = self.cn_dataset_schema

        self.general_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.SUBSET_ASCENDABLE_SCHEMA,
            self.DB_SUBSET_ASCENDABLE_SCHEMA if support_dreambooth else {},
            self.CN_SUBSET_ASCENDABLE_SCHEMA if support_controlnet else {},
            self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
        )

        self.user_config_validator = Schema(
            {
                "general": self.general_schema,
                "datasets": [self.dataset_schema],
            }
        )

        self.argparse_schema = self.__merge_dict(
            self.general_schema,
            self.ARGPARSE_SPECIFIC_SCHEMA,
            {
                optname: Any(None, self.general_schema.get(optname, None))
                for optname in self.ARGPARSE_NULLABLE_OPTNAMES
            },
            {
                a_name: self.general_schema.get(c_name, None)
                for a_name, c_name in self.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME.items()
            },
        )

        self.argparse_config_validator = Schema(
            Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA
        )

    def sanitize_user_config(self, user_config: dict) -> dict:
        try:
            return self.user_config_validator(user_config)
        except MultipleInvalid:
            print("Invalid user config / ユーザ設定の形式が正しくないようです")
            raise

    # NOTE: In nature, argument parser result is not needed to be sanitize
    #   However this will help us to detect program bug
    def sanitize_argparse_namespace(
        self, argparse_namespace: argparse.Namespace
    ) -> argparse.Namespace:
        try:
            return self.argparse_config_validator(argparse_namespace)
        except MultipleInvalid:
            # Found a bug
            print(
                "Invalid cmdline parsed arguments. This should be a bug. / コマンドラインのパース結果が正しくないようです。プログラムのバグの可能性が高いです。"
            )
            raise

    # NOTE: value would be overwritten by latter dict if there is already the same key
    @staticmethod
    def __merge_dict(*dict_list: dict) -> dict:
        merged = {}
        for schema in dict_list:
            # merged |= schema
            for k, v in schema.items():
                merged[k] = v
        return merged


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}

    def __init__(self, sanitizer: ConfigSanitizer):
        self.sanitizer = sanitizer

    # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
    def generate(
        self,
        user_config: dict,
        argparse_namespace: argparse.Namespace,
        **runtime_params,
    ) -> Blueprint:
        sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(
            argparse_namespace
        )

        # convert argparse namespace to dict like config
        # NOTE: it is ok to have extra entries in dict
        optname_map = self.sanitizer.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME
        argparse_config = {
            optname_map.get(optname, optname): value
            for optname, value in vars(sanitized_argparse_namespace).items()
        }

        general_config = sanitized_user_config.get("general", {})

        dataset_blueprints = []
        for dataset_config in sanitized_user_config.get("datasets", []):
            # NOTE: if subsets have no "metadata_file", these are DreamBooth datasets/subsets
            subsets = dataset_config.get("subsets", [])
            is_dreambooth = all(["metadata_file" not in subset for subset in subsets])
            is_controlnet = False
            if is_controlnet:
                raise NotImplementedError("Not support now")
            elif is_dreambooth:
                subset_params_klass = DreamBoothSubsetParams
                dataset_params_klass = DreamBoothDatasetParams
            else:
                raise NotImplementedError("Not support now")

            subset_blueprints = []
            for subset_config in subsets:
                params = self.generate_params_by_fallbacks(
                    subset_params_klass,
                    [
                        subset_config,
                        dataset_config,
                        general_config,
                        argparse_config,
                        runtime_params,
                    ],
                )
                subset_blueprints.append(SubsetBlueprint(params))

            params = self.generate_params_by_fallbacks(
                dataset_params_klass,
                [dataset_config, general_config, argparse_config, runtime_params],
            )
            dataset_blueprints.append(
                DatasetBlueprint(
                    is_dreambooth, is_controlnet, params, subset_blueprints
                )
            )

        dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

        return Blueprint(dataset_group_blueprint)

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
        name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        search_value = BlueprintGenerator.search_value
        default_params = asdict(param_klass())
        param_names = default_params.keys()

        params = {
            name: search_value(
                name_map.get(name, name), fallbacks, default_params.get(name)
            )
            for name in param_names
        }

        return param_klass(**params)

    @staticmethod
    def search_value(key: str, fallbacks: Sequence[dict], default_value=None):
        for cand in fallbacks:
            value = cand.get(key)
            if value is not None:
                return value

        return default_value


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible

    resos_list = list(resos)
    resos_list.sort()
    return resos_list


def generate_dataset_group_by_blueprint(dataset_group_blueprint: DatasetGroupBlueprint):
    datasets: List[DreamBoothDataset] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        if dataset_blueprint.is_controlnet:
            raise NotImplementedError("Not support now")
        elif dataset_blueprint.is_dreambooth:
            subset_klass = DreamBoothSubset
            dataset_klass = DreamBoothDataset
        else:
            raise NotImplementedError("Not support now")

        subsets = [
            subset_klass(**asdict(subset_blueprint.params))
            for subset_blueprint in dataset_blueprint.subsets
        ]
        dataset = dataset_klass(subsets=subsets, **asdict(dataset_blueprint.params))
        datasets.append(dataset)

    # print info
    info = ""
    for i, dataset in enumerate(datasets):
        is_dreambooth = isinstance(dataset, DreamBoothDataset)
        is_controlnet = False
        info += dedent(
            f"""\
                [Dataset {i}]
                    batch_size: {dataset.batch_size}
                    resolution: {(dataset.width, dataset.height)}
                    enable_bucket: {dataset.enable_bucket}
                    network_multiplier: {dataset.network_multiplier}
                """
        )

        if dataset.enable_bucket:
            info += indent(
                dedent(
                    f"""\
                        min_bucket_reso: {dataset.min_bucket_reso}
                        max_bucket_reso: {dataset.max_bucket_reso}
                        bucket_reso_steps: {dataset.bucket_reso_steps}
                        bucket_no_upscale: {dataset.bucket_no_upscale}
                    \n"""
                ),
                "  ",
            )
        else:
            info += "\n"

        for j, subset in enumerate(dataset.subsets):
            info += indent(
                dedent(
                    f"""\
                        [Subset {j} of Dataset {i}]
                        image_dir: "{subset.image_dir}"
                        image_count: {subset.img_count}
                        num_repeats: {subset.num_repeats}
                        shuffle_caption: {subset.shuffle_caption}
                        keep_tokens: {subset.keep_tokens}
                        keep_tokens_separator: {subset.keep_tokens_separator}
                        caption_dropout_rate: {subset.caption_dropout_rate}
                        caption_dropout_every_n_epoches: {subset.caption_dropout_every_n_epochs}
                        caption_tag_dropout_rate: {subset.caption_tag_dropout_rate}
                        caption_prefix: {subset.caption_prefix}
                        caption_suffix: {subset.caption_suffix}
                        color_aug: {subset.color_aug}
                        flip_aug: {subset.flip_aug}
                        face_crop_aug_range: {subset.face_crop_aug_range}
                        random_crop: {subset.random_crop}
                        token_warmup_min: {subset.token_warmup_min},
                        token_warmup_step: {subset.token_warmup_step},
                    """
                ),
                "  ",
            )

            if is_dreambooth:
                info += indent(
                    dedent(
                        f"""\
                            is_reg: {subset.is_reg}
                            class_tokens: {subset.class_tokens}
                            caption_extension: {subset.caption_extension}
                            \n"""
                    ),
                    "    ",
                )
            elif not is_controlnet:
                info += indent(
                    dedent(
                        f"""\
                            metadata_file: {subset.metadata_file}
                            \n"""
                    ),
                    "    ",
                )

    print(info)

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no
    for i, dataset in enumerate(datasets):
        print(f"[Dataset {i}]")
        dataset.make_buckets()
        dataset.set_seed(seed)

    return DatasetGroup(datasets)


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(
                glob.glob(os.path.join(glob.escape(directory), base + ext))
            )
        else:
            img_paths.extend(
                glob.glob(glob.escape(os.path.join(directory, base + ext)))
            )
    img_paths = list(set(img_paths))
    img_paths.sort()
    return img_paths


def load_tokenizer(args: argparse.Namespace):
    print("prepare tokenizer")
    original_path = V2_STABLE_DIFFUSION_PATH if args.v2 else TOKENIZER_PATH

    tokenizer: CLIPTokenizer = None
    if args.tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(
            args.tokenizer_cache_dir, original_path.replace("/", "_")
        )
        if os.path.exists(local_tokenizer_path):
            print(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer = CLIPTokenizer.from_pretrained(
                local_tokenizer_path
            )  # same for v1 and v2

    if tokenizer is None:
        if args.v2:
            tokenizer = CLIPTokenizer.from_pretrained(
                original_path, subfolder="tokenizer"
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        print(f"update token length: {args.max_token_length}")

    if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
        print(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer


def generate_dreambooth_subsets_config_by_subdirs(
    train_data_dir: Optional[str] = None, reg_data_dir: Optional[str] = None
):
    def extract_dreambooth_params(name: str) -> Tuple[int, str]:
        tokens = name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError as e:
            print(
                f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {name}"
            )
            return 0, ""
        caption_by_folder = "_".join(tokens[1:])
        return n_repeats, caption_by_folder

    def generate(base_dir: Optional[str], is_reg: bool):
        if base_dir is None:
            return []

        base_dir: Path = Path(base_dir)
        if not base_dir.is_dir():
            return []

        subsets_config = []
        for subdir in base_dir.iterdir():
            if not subdir.is_dir():
                continue

            num_repeats, class_tokens = extract_dreambooth_params(subdir.name)
            if num_repeats < 1:
                continue

            subset_config = {
                "image_dir": str(subdir),
                "num_repeats": num_repeats,
                "is_reg": is_reg,
                "class_tokens": class_tokens,
            }
            subsets_config.append(subset_config)

        if subsets_config == []:
            subset_config = {
                "image_dir": str(base_dir),
                "num_repeats": 1,
                "is_reg": is_reg,
                "class_tokens": str(base_dir),
            }
            subsets_config.append(subset_config)
        return subsets_config

    subsets_config = []
    subsets_config += generate(train_data_dir, False)
    subsets_config += generate(reg_data_dir, True)

    return subsets_config
