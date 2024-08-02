# Copyright (c) 2023 DeepFloyd; StabilityAI

import html
import re
import string
import urllib.parse as ul

import decord
import numpy as np
import pandas as pd
import torch
import torchvision
from diffusers.utils import is_bs4_available, is_ftfy_available
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from mindspeed_mm.data.data_utils.data_transform import TemporalRandomCrop
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms

if is_bs4_available():
    from bs4 import BeautifulSoup
if is_ftfy_available():
    import ftfy

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class GetDataFromPath:
    """get the data from different types of files such as csv/json/parquat"""

    def __init__(self, input_path):
        self.data_path = input_path

    def __call__(self):
        return self.get_datasamples()

    def get_datasamples(self):
        if self.data_path.endswith(".csv"):
            data_out = pd.read_csv(self.data_path)
            return data_out.to_dict("records")
        elif self.data_path.endswith(".json"):
            data_out = pd.read_json(self.data_path)
            return data_out.to_dict("records")
        elif self.data_path.endswith(".parquat"):
            data_out = pd.read_parquat(self.data_path)
            return data_out.to_dict("records")
        else:
            raise NotImplementedError(f"Unsupported file format: {self.data_path}")


class DecordInit:
    """Using Decord to initialize the video_reader."""

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
        **kwargs,
    ):
        self.num_frames = num_frames
        self.video_transforms = get_transforms(
            is_video=True, train_pipeline=train_pipeline
        )
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)

    def __call__(self, vframes, is_decord_read=False):
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
        return video


class ImageProcesser:
    """Used for image data preprocessing"""

    def __init__(
        self,
        num_frames=16,
        train_pipeline=None,
        image_reader_type="torchvision",
        **kwargs,
    ):
        self.num_frames = num_frames
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline
        )
        self.image_reader_type = image_reader_type

    def __call__(self, image_path):
        video = self.image_to_video(image_path)
        return video

    def image_to_video(self, image_path):
        image = self.image_reader(image_path)
        image = self.transform(image)
        video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        video = video.permute(1, 0, 2, 3)  # TCHW -> CTHW
        return video

    def image_reader(self, image_path):
        if self.image_reader_type == "torchvision":
            image = pil_loader(image_path)
        else:
            raise NotImplementedError(
                f"Unsupported image reader type: {self.image_reader_type}"
            )
        return image


class TextProcesser:
    """Used for text data preprocessing"""

    def __init__(self, model_max_length, tokenizer, padding_type="max_length"):
        self.model_max_length = model_max_length
        self.padding = padding_type
        self.tokenizer = tokenizer
        # raw processing: r'[' + '#®•©™&@·º½¾¿¡§~' + '\)' + '\(' + '\]' + '\[' + '\}' + '\{' + '\|' + '\\' + '\/' + '\*' + r']{1,}')
        self.bad_punct_regex = re.compile(
            r"[" + re.escape(string.punctuation) + "]{1,}"
        )  # noqa

    def __call__(self, text, vid_img_fusion_by_splicing):
        text_info = self.text_preprocessing(text[0])
        text_tokens_and_mask = self.tokenizer(
            text_info,
            max_length=self.model_max_length,
            padding=self.padding,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"].squeeze(0)
        cond_mask = text_tokens_and_mask["attention_mask"].squeeze(0)
        return input_ids, cond_mask

    def text_preprocessing(self, text):
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text

    def basic_clean(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
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
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
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
            self.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

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


# TODO
def get_batch_on_this_tp_rank(data_iterator):
    """
    :param data_iterator:
    :return:
    """
    # Note: not support now
    batch = data_iterator
    return batch
