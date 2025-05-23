# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import random
from typing import Union
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    PROMPT_MASK_2,
    PROMPT_IDS_2,
    TEXT,
    VIDEO,
    IMG_FPS
)
from mindspeed_mm.data.data_utils.utils import (
    VID_EXTENSIONS,
    ImageProcesser,
    TextProcesser,
    VideoProcesser,
    VideoReader
)
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models import Tokenizer
from mindspeed_mm.data.data_utils.data_transform import (
    MaskGenerator,
    add_aesthetic_notice_image,
    add_aesthetic_notice_video
)


T2VOutputData = {
    VIDEO: [],
    PROMPT_IDS: [],
    PROMPT_MASK: [],
    PROMPT_IDS_2: [],
    PROMPT_MASK_2: [],
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
        enable_text_preprocessing: bool = True,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        model_max_length: int = 120,
        tokenizer_config: Union[dict, None] = None,
        tokenizer_config_2: Union[dict, None] = None,
        use_feature_data: bool = False,
        use_img_from_vid: bool = True,
        **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.enable_text_preprocessing = enable_text_preprocessing
        self.use_feature_data = use_feature_data
        self.use_img_from_vid = use_img_from_vid

        self.num_frames = vid_img_process.get("num_frames", 16)
        self.frame_interval = vid_img_process.get("frame_interval", 1)

        self.max_height = vid_img_process.get("max_height", 480)
        self.max_width = vid_img_process.get("max_width", 640)
        self.max_hxw = vid_img_process.get("max_hxw", None)
        self.min_hxw = vid_img_process.get("min_hxw", None)
        self.train_fps = vid_img_process.get("train_fps", 24)
        self.speed_factor = vid_img_process.get("speed_factor", 1.0)
        self.too_long_factor = vid_img_process.get("too_long_factor", 5.0)
        self.drop_short_ratio = vid_img_process.get("drop_short_ratio", 1.0)
        self.cfg = vid_img_process.get("cfg", 0.1)
        self.image_processer_type = vid_img_process.get(
            "image_processer_type", "image2image"
        )
        self.hw_stride = vid_img_process.get("hw_stride", 16)
        self.ae_stride_t = vid_img_process.get("ae_stride_t", 8)
        self.force_resolution = vid_img_process.get("force_resolution", True)
        self.force_5_ratio = vid_img_process.get("force_5_ratio", False)
        self.sp_size = vid_img_process.get("sp_size", 1)
        self.train_sp_batch_size = vid_img_process.get("train_sp_batch_size", 1)
        self.gradient_accumulation_size = vid_img_process.get("gradient_accumulation_size", 1)
        self.batch_size = vid_img_process.get("batch_size", 1)
        self.seed = vid_img_process.get("seed", 42)
        self.max_h_div_w_ratio = vid_img_process.get("max_h_div_w_ratio", 2.0)
        self.min_h_div_w_ratio = vid_img_process.get("min_h_div_w_ratio", 0.5)
        self.min_num_frames = vid_img_process.get("min_num_frames", 29)
        self.use_aesthetic = vid_img_process.get("use_aesthetic", False) 

        max_workers = vid_img_process.get("max_workers", 1)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout = vid_img_process.get("timeout", 600) 
        
        if self.max_hxw is not None and self.min_hxw is None:
            self.min_hxw = self.max_hxw // 4
        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        self.video_reader_type = vid_img_process.get("video_reader_type", "decoder")
        self.image_reader_type = vid_img_process.get("image_reader_type", "Image")
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.video_processer = VideoProcesser(
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
            data_storage_mode=self.data_storage_mode,
            train_fps=self.train_fps,
            speed_factor=self.speed_factor,
            too_long_factor=self.too_long_factor,
            drop_short_ratio=self.drop_short_ratio,
            max_height=self.max_height,
            max_width=self.max_width,
            max_hxw=self.max_hxw,
            min_hxw=self.min_hxw,
            force_resolution=self.force_resolution,
            force_5_ratio=self.force_5_ratio,
            seed=self.seed,
            hw_stride=self.hw_stride,
            max_h_div_w_ratio=self.max_h_div_w_ratio,
            min_h_div_w_ratio=self.min_h_div_w_ratio,
            ae_stride_t=self.ae_stride_t,
            sp_size=self.sp_size,
            train_sp_batch_size=self.train_sp_batch_size,
            gradient_accumulation_size=self.gradient_accumulation_size,
            batch_size=self.batch_size,
            min_num_frames=self.min_num_frames
        )
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
            image_processer_type=self.image_processer_type,
        )
        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.tokenizer_2 = None
            if tokenizer_config_2 is not None:
                self.tokenizer_2 = Tokenizer(tokenizer_config_2).get_tokenizer()
            self.text_processer = TextProcesser(
                model_max_length=model_max_length,
                tokenizer=self.tokenizer,
                enable_text_preprocessing=self.enable_text_preprocessing,
                tokenizer_2=self.tokenizer_2,
                use_clean_caption=use_clean_caption,
                support_chinese=support_chinese,
                cfg=self.cfg,
            )

        if self.data_storage_mode == "combine":
            self.data_samples, self.sample_size, self.shape_idx_dict = (
                self.video_processer.define_frame_index(self.data_samples)
            )
            self.lengths = self.sample_size

    def __getitem__(self, index):
        try:
            future = self.executor.submit(self.getitem, index)
            data = future.result(timeout=self.timeout) 
            return data
        except Exception as e:
            if len(str(e)) < 2:
                e = f"TimeoutError, {self.timeout}s timeout occur with {self.data_samples.iloc[index]['path']}"
            print(f"Error: {e}")
            index_cand = self.shape_idx_dict[self.sample_size[index]]  # pick same shape
            return self.__getitem__(random.choice(index_cand))

    def __len__(self):
        return len(self.data_samples)

    def getitem(self, index):
        # init output data
        examples = T2VOutputData
        if self.data_storage_mode == "combine":
            examples = self.get_merge_data(examples, index)
        else:
            raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode}."
            )
        return examples

    def get_data_from_feature_data(self, sample):
        raise NotImplementedError("Not implemented.")

    def get_merge_data(self, examples, index):
        sample = self.data_samples.iloc[index]
        file_path = sample["path"]
        if not os.path.exists(file_path):
            raise AssertionError(f"file {file_path} do not exist!")
        file_type = self.get_type(file_path)
        if file_type == "video":
            predefine_frame_indice = sample["sample_frame_index"]
            start_frame_idx = sample["start_frame_idx"]
            clip_total_frames = sample["num_frames"]
            fps = sample["fps"]
            crop = sample.get("crop", [None, None, None, None])
            vframes, _, is_decord_read = self.video_reader(file_path)
            video = self.video_processer(
                vframes,
                is_decord_read=is_decord_read,
                start_frame_idx=start_frame_idx,
                clip_total_frames=clip_total_frames,
                predefine_frame_indice=predefine_frame_indice,
                fps=fps,
                crop=crop,
            )
            examples[VIDEO] = video
        elif file_type == "image":
            image = self.image_processer(file_path)
            examples[VIDEO] = image

        text = sample["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if self.use_aesthetic:
            if sample.get('aesthetic', None) is not None or sample.get('aes', None) is not None:
                aes = sample.get('aesthetic', None) or sample.get('aes', None)
                if file_type == "video":
                    text = [add_aesthetic_notice_video(text[0], aes)]
                elif file_type == "image":
                    text = [add_aesthetic_notice_image(text[0], aes)]
        prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.get_text_processer(text)# tokenizer, tokenizer_2
        examples[PROMPT_IDS], examples[PROMPT_MASK], examples[PROMPT_IDS_2], examples[PROMPT_MASK_2] = prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2
        return examples

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.text_processer(texts)# tokenizer, tokenizer_2
        return (prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2)


class DynamicVideoTextDataset(MMBaseDataset):
    """
    A mutilmodal dataset for variable text-to-video task based on MMBaseDataset

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
        enable_text_preprocessing: bool = True,
        use_clean_caption: bool = True,
        model_max_length: int = 120,
        tokenizer_config: Union[dict, None] = None,
        use_feature_data: bool = False,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        dummy_text_feature=False,
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
        
        if "video_mask_ratios" in kwargs:
            self.video_mask_generator = MaskGenerator(kwargs["video_mask_ratios"])
        else:
            self.video_mask_generator = None

        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.text_processer = TextProcesser(
                model_max_length=model_max_length,
                tokenizer=self.tokenizer,
                use_clean_caption=use_clean_caption,
                enable_text_preprocessing=enable_text_preprocessing
            )

        self.data_samples["id"] = np.arange(len(self.data_samples))
        self.dummy_text_feature = dummy_text_feature
        self.get_text = "text" in self.data_samples.columns

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]

    def get_value_from_vid_or_img(self, num_frames, video_or_image_path, image_size):
        file_type = self.get_type(video_or_image_path)

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, vinfo, _ = self.video_reader(video_or_image_path)
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            video_fps = video_fps // self.frame_interval

            video = self.video_processer(vframes, num_frames=num_frames, frame_interval=self.frame_interval,
                                         image_size=image_size)  # T C H W
        else:
            # loading
            image = pil_loader(video_or_image_path)
            video_fps = IMG_FPS

            # transform
            image = self.image_processer(image)

            # repeat
            video = image.unsqueeze(0)

        return video, video_fps

    def __getitem__(self, index):
        index, num_frames, height, width = [int(val) for val in index.split("-")]
        sample = self.data_samples.iloc[index]
        video_or_image_path = sample["path"]
        if self.data_folder:
            video_or_image_path = os.path.join(self.data_folder, video_or_image_path)
            
        video, video_fps = self.get_value_from_vid_or_img(num_frames, video_or_image_path, image_size=(height, width))
        ar = height / width

        ret = {
            "video": video,
            "video_mask": None,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        
        if self.video_mask_generator is not None:
            ret["video_mask"] = self.video_mask_generator.get_mask(video)

        if self.get_text:
            prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.get_text_processer(sample["text"])# tokenizer, tokenizer_2
            ret["prompt_ids"] = prompt_ids
            ret["prompt_mask"] = prompt_mask

        if self.dummy_text_feature:
            text_len = 50
            ret["prompt_ids"] = torch.zeros((1, text_len, 1152))
            ret["prompt_mask"] = text_len

        return ret

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.text_processer(texts)
        return prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2
