# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
from typing import Union

import numpy as np
import torch
import torchvision

from mindspeed_mm.data.data_utils.utils import (
    VID_EXTENSIONS,
    GetDataFromPath,
    ImageProcesser,
    TextProcesser,
    VideoProcesser,
    VideoReader,
)
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models import Tokenizer

T2VOutputData = {
    "video_v": None,
    "video_f": None,
    "text": None,
    "input_ids": None,
    "cond_mask": None,
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
        model_max_length: int = 1,
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
        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        self.video_reader_type = vid_img_process.get("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.get("image_reader_type", "torchvision")
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.video_processer = VideoProcesser(
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
        )
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
        )
        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.text_processer = TextProcesser(model_max_length=model_max_length)

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            path = self.data_samples[index]["path"]
            print(f"Data {path}: the error is {e}")
            return self.getitem(np.random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_samples)

    def getitem(self, index):
        # init output data:
        examples = T2VOutputData
        sample = self.data_samples[index]
        if self.use_feature_data:
            raise NotImplementedError("Not support now.")
        else:
            path, text = sample["path"], sample["cap"]
            if self.data_folder:
                path = os.path.join(self.data_path, path)
            examples["text"] = text
            video_value = (
                self.get_vid_img_fusion(path)
                if self.vid_img_fusion_by_splicing
                else self.get_value_from_vid_or_img(path)
            )
            examples["video_v"] = video_value
            if self.use_text_processer:
                input_ids, cond_mask = self.get_text_processer(text)
                examples["input_ids"], examples["cond_mask"] = input_ids, cond_mask
        return examples

    # TODO: support soon
    def get_data_from_feature_data(self, sample):
        return sample

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

    def get_text_processer(self, text):
        input_ids, cond_mask = self.text_processer(text, self.tokenizer)
        if self.vid_img_fusion_by_splicing and self.use_img_from_vid:
            input_ids = torch.stack(
                [input_ids] * (1 + self.use_image_num)
            )  # 1+self.use_image_num, l
            cond_mask = torch.stack(
                [cond_mask] * (1 + self.use_image_num)
            )  # 1+self.use_image_num, l
        if self.vid_img_fusion_by_splicing and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")
        return input_ids, cond_mask


class VariableT2VDataset(T2VDataset):
    pass
