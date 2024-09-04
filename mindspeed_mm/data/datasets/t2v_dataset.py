# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import random
from collections import Counter
from typing import Union

import numpy as np
import torch
import torchvision

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    TEXT,
    VIDEO,
)
from mindspeed_mm.data.data_utils.data_transform import TemporalRandomCrop
from mindspeed_mm.data.data_utils.utils import (
    VID_EXTENSIONS,
    DataSetProg,
    ImageProcesser,
    TextProcesser,
    VideoProcesser,
    VideoReader,
)
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models import Tokenizer

T2VOutputData = {
    VIDEO: [],
    TEXT: [],
    PROMPT_IDS: [],
    PROMPT_MASK: [],
}


class T2VDataset(MMBaseDataset):
    """
    A mutilmodal dataset for text-to-video task based on MMBaseDataset

    Args: some parameters from dataset_param_dict in config.
        basic_param(dict): some basic parameters such as data_path, data_folder, etc.
        vid_img_process(dict): some data preprocessing parameters
        use_text_processer(bool): whether text preprocessing
        tokenizer_config(dict): the config of tokenizer
        use_feature_data(bool): use vae feature instead of raw video data or use text feature instead of raw text.
        vid_img_fusion_by_splicing(bool):  videos and images are fused by splicing
        use_img_num(int): the number of fused images
        use_img_from_vid(bool): sampling some images from video
    """

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        model_max_length: int = 120,
        tokenizer_config: Union[dict, None] = None,
        use_feature_data: bool = False,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.use_feature_data = use_feature_data
        self.vid_img_fusion_by_splicing = vid_img_fusion_by_splicing
        self.use_img_num = use_img_num
        self.use_img_from_vid = use_img_from_vid

        self.num_frames = vid_img_process.get("num_frames", 16)
        self.frame_interval = vid_img_process.get("frame_interval", 1)
        self.resolution = vid_img_process.get("resolution", (256, 256))

        self.max_height = vid_img_process.get("max_height", 480)
        self.max_width = vid_img_process.get("max_width", 640)
        self.train_fps = vid_img_process.get("train_fps", 24)
        self.speed_factor = vid_img_process.get("speed_factor", 1.0)
        self.drop_short_ratio = vid_img_process.get("drop_short_ratio", 1.0)
        self.cfg = vid_img_process.get("cfg", 0.1)
        self.image_processer_type = vid_img_process.get(
            "image_processer_type", "image2video"
        )

        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        self.video_reader_type = vid_img_process.get("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.get("image_reader_type", "torchvision")
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.video_processer = VideoProcesser(
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
            data_storage_mode=self.data_storage_mode,
            train_fps=self.train_fps,
            speed_factor=self.speed_factor,
            drop_short_ratio=self.drop_short_ratio,
            max_height=self.max_height,
            max_width=self.max_width,
        )
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
            image_processer_type=self.image_processer_type,
        )
        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.text_processer = TextProcesser(
                model_max_length=model_max_length,
                tokenizer=self.tokenizer,
                use_clean_caption=use_clean_caption,
                support_chinese=support_chinese,
                cfg=self.cfg,
            )

        if self.data_storage_mode == "combine":
            self.dataset_prog = DataSetProg()
            dataloader_num_workers = vid_img_process.get("dataloader_num_workers", 1)
            self.data_samples, self.sample_num_frames = (
                self.video_processer.define_frame_index(self.data_samples)
            )
            self.lengths = self.sample_num_frames
            n_elements = len(self.data_samples)
            self.dataset_prog.set_cap_list(
                dataloader_num_workers, self.data_samples, n_elements
            )

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            if self.data_storage_mode == "standard":
                path = self.data_samples[index][FILE_INFO]
                print(f"Data {path}: the error is {e}")
            else:
                print(f"the error is {e}")
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_samples)

    def getitem(self, index):
        # init output data
        examples = T2VOutputData
        if self.data_storage_mode == "standard":
            sample = self.data_samples[index]
            if self.use_feature_data:
                raise NotImplementedError("Not support now.")
            else:
                path, texts = sample[FILE_INFO], sample[CAPTIONS]
                if self.data_folder:
                    path = os.path.join(self.data_folder, path)
                examples[TEXT] = texts
                video_value = (
                    self.get_vid_img_fusion(path)
                    if self.vid_img_fusion_by_splicing
                    else self.get_value_from_vid_or_img(path)
                )
                examples[VIDEO] = video_value
                if self.use_text_processer:
                    prompt_ids, prompt_mask = self.get_text_processer(texts)
                    examples[PROMPT_IDS], examples[PROMPT_MASK] = (
                        prompt_ids,
                        prompt_mask,
                    )
        elif self.data_storage_mode == "combine":
            examples = self.get_merge_data(examples, index)
        else:
            raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode}."
            )
        return examples

    # TODO: support soon
    def get_data_from_feature_data(self, sample):
        return sample

    def get_merge_data(self, examples, index):
        sample = self.dataset_prog.cap_list[index]
        file_path = sample["path"]
        if not os.path.exists(file_path):
            raise AssertionError(f"file {file_path} do not exist!")
        file_type = self.get_type(file_path)
        if file_type == "video":
            frame_indice = sample["sample_frame_index"]
            vframes, is_decord_read = self.video_reader(file_path)
            video = self.video_processer(
                vframes,
                is_decord_read=is_decord_read,
                predefine_num_frames=len(frame_indice),
            )
            examples[VIDEO] = video
        elif file_type == "image":
            image = self.image_processer(file_path)
            examples[VIDEO] = image

        text = sample["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        prompt_ids, prompt_mask = self.get_text_processer(text)
        examples[PROMPT_IDS], examples[PROMPT_MASK] = prompt_ids, prompt_mask
        return examples

    def get_value_from_vid_or_img(self, path):
        file_type = self.get_type(path)
        if file_type == "video":
            vframes, is_decord_read = self.video_reader(path)
            video_value = self.video_processer(vframes, is_decord_read)
        elif file_type == "image":
            video_value = self.image_processer(path)
        return video_value

    def get_vid_img_fusion(self, path):
        vframes, is_decord_read = self.video_reader(path)
        video_value = self.video_processer(vframes, is_decord_read)
        if self.use_img_num != 0 and self.use_img_from_vid:
            select_image_idx = np.linspace(
                0, self.num_frames - 1, self.use_img_num, dtype=int
            )
            if self.num_frames < self.use_image_num:
                raise AssertionError(
                    "The num_frames must be larger than the use_image_num."
                )
            images = video_value[:, select_image_idx]  # c, num_img, h, w
            video_value = torch.cat(
                [video_value, images], dim=1
            )  # c, num_frame+num_img, h, w
            return video_value
        elif self.use_img_num != 0 and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")
        else:
            raise NotImplementedError

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask = self.text_processer(texts)
        if self.vid_img_fusion_by_splicing and self.use_img_from_vid:
            prompt_ids = torch.stack(
                [prompt_ids] * (1 + self.use_image_num)
            )  # 1+self.use_image_num, l
            prompt_mask = torch.stack(
                [prompt_mask] * (1 + self.use_image_num)
            )  # 1+self.use_image_num, l
        if self.vid_img_fusion_by_splicing and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")
        return prompt_ids, prompt_mask


class VariableT2VDataset(T2VDataset):
    pass