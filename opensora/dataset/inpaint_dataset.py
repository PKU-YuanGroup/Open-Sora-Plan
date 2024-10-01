import time
import traceback

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import glob
import json
import pickle
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from os.path import join as opj
from collections import Counter

import cv2
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger
import gc

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from opensora.dataset.transform import get_params, longsideresize, add_masking_notice, motion_mapping_fun, calculate_statistics, \
    add_webvid_watermark_notice, clean_vidal, add_high_aesthetic_notice_image, add_aesthetic_notice_video, add_high_aesthetic_notice_image_human

from opensora.utils.mask_utils import MaskProcessor, STR_TO_TYPE
from opensora.dataset.t2v_datasets import T2V_dataset, DataSetProg

logger = get_logger(__name__)

dataset_prog = DataSetProg()

def type_ratio_normalize(mask_type_ratio_dict):
    for k, v in mask_type_ratio_dict.items():
        assert v >= 0, f"mask_type_ratio_dict[{k}] should be non-negative, but got {v}"
    total = sum(mask_type_ratio_dict.values())
    length = len(mask_type_ratio_dict)
    if total == 0:
        return {k: 1.0 / length for k in mask_type_ratio_dict.keys()}
    return {k: v / total for k, v in mask_type_ratio_dict.items()}

class Inpaint_dataset(T2V_dataset):
    def __init__(self, args, resize_transform, transform, resize_transform_img, temporal_sample, tokenizer_1, tokenizer_2):
        super().__init__(
            args=args, 
            transform=transform, 
            transform_img=transform, 
            temporal_sample=temporal_sample, 
            tokenizer_1=tokenizer_1, 
            tokenizer_2=tokenizer_2
        )

        self.resize_transform = resize_transform
        self.resize_transform_img = resize_transform_img

        if self.num_frames != 1:
            self.mask_type_ratio_dict_video = args.mask_type_ratio_dict_video if args.mask_type_ratio_dict_video is not None else {'random_temporal': 1.0}
            self.mask_type_ratio_dict_video = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_video.items()}
            self.mask_type_ratio_dict_video = type_ratio_normalize(self.mask_type_ratio_dict_video)
                
        self.mask_type_ratio_dict_image = args.mask_type_ratio_dict_image if args.mask_type_ratio_dict_image is not None else {'random_spatial': 1.0}
        self.mask_type_ratio_dict_image = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_image.items()}
        self.mask_type_ratio_dict_image = type_ratio_normalize(self.mask_type_ratio_dict_image)

        print(f"mask_type_ratio_dict_video: {self.mask_type_ratio_dict_video}")
        print(f"mask_type_ratio_dict_image: {self.mask_type_ratio_dict_image}")

        self.mask_processor = MaskProcessor(
            max_height=args.max_height,
            max_width=args.max_width,
            min_clear_ratio=args.min_clear_ratio,
            max_clear_ratio=args.max_clear_ratio,
            min_clear_ratio_hw=args.min_clear_ratio_hw,
            max_clear_ratio_hw=args.max_clear_ratio_hw,
            dilate_kernel_size=args.dilate_kernel_size,
        )

        self.default_text_ratio = args.default_text_ratio

    def drop(self, text, is_video=True):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            if rand_num_text < self.default_text_ratio:
                if not is_video:
                    text = "The image showcases a scene with coherent and clear visuals." 
                else:
                    text = "The video showcases a scene with coherent and clear visuals." 
            else:
                text = ''

        return dict(text=text)

    def get_video(self, idx):
        # npu_config.print_msg(f"current idx is {idx}")
        # video = random.choice([random_video_noise(65, 3, 336, 448), random_video_noise(65, 3, 1024, 1024), random_video_noise(65, 3, 360, 480)])
        # # print('random shape', video.shape)
        # input_ids = torch.ones(1, 120).to(torch.long).squeeze(0)
        # cond_mask = torch.cat([torch.ones(1, 60).to(torch.long), torch.ones(1, 60).to(torch.long)], dim=1).squeeze(0)
        # logger.info(f'Now we use t2v dataset {idx}')
        video_data = dataset_prog.cap_list[idx]
        video_path = video_data['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']
        
        mask = None

        if self.video_reader == 'decord':
            video = self.decord_read(video_path, predefine_frame_indice=frame_indice)
            if video_data.get('mask_path', None) is not None:
                mask = self.decord_read(video_data['mask_path'], predefine_frame_indice=frame_indice)
        elif self.video_reader == 'opencv':
            video = self.opencv_read(video_path, predefine_frame_indice=frame_indice)
            if video_data.get('mask_path', None) is not None:
                mask = self.opencv_read(video_data['mask_path'], predefine_frame_indice=frame_indice)
        else:
            NotImplementedError(f'Found {self.video_reader}, but support decord or opencv')
        # import ipdb;ipdb.set_trace()

        video = self.resize_transform(video)  # T C H W -> T C H W

        # binary mask
        if mask is not None:
            mask = mask[:, 0:1]
            mask = self.resize_transform(mask)  # T C H W -> T C H W
            mask = (mask > 128).to(dtype=video.dtype, device=video.device)
        inpaint_cond_data = self.mask_processor(video, mask=mask, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        video = self.transform(video)  # T C H W -> T C H W
        masked_video = self.transform(masked_video)  # T C H W -> T C H W
        assert video.shape[2] == sample_h and video.shape[3] == sample_w, f'sample_h ({sample_h}), sample_w ({sample_w}), video ({video.shape})'

        video = torch.cat([video, masked_video, mask], dim=1)  # T 2C+1 H W

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = video_data['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if '/VIDAL-10M/' in video_path:
            text = [clean_vidal(text[0])]
        if '/Webvid-10M/' in video_path:
            text = [add_webvid_watermark_notice(text[0])]
        if not (video_data.get('aesthetic', None) is None):
            text = [add_aesthetic_notice_video(text[0], video_data['aesthetic'])]

        text = [text[0].replace(' image ', ' video ').replace(' image,', ' video,')]
        text = text_preprocessing(text, support_Chinese=self.support_Chinese)
        text = self.drop(text, is_video=True)['text']

        text_tokens_and_mask_1 = self.tokenizer_1(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_1 = text_tokens_and_mask_1['input_ids']
        cond_mask_1 = text_tokens_and_mask_1['attention_mask']
        
        input_ids_2, cond_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids_2 = text_tokens_and_mask_2['input_ids']
            cond_mask_2 = text_tokens_and_mask_2['attention_mask']

        if self.use_motion:
            motion_score = motion_mapping_fun(video_data['motion_score'])
            return dict(
                pixel_values=video, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, motion_score=motion_score, 
                input_ids_2=input_ids_2, cond_mask_2=cond_mask_2,
                )
        else:
            return dict(
                pixel_values=video, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, motion_score=None, 
                input_ids_2=input_ids_2, cond_mask_2=cond_mask_2,
                )

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        sample_h = image_data['resolution']['sample_height']
        sample_w = image_data['resolution']['sample_width']
        is_ood_img =  image_data['is_ood_img']

        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]

        mask = None
        if image_data.get('mask_path', None) is not None:
            mask = Image.open(image_data['mask_path']).convert('L')

        if is_ood_img:
            image = self.resize_transform_img(image)
        else:
            image = self.resize_transform(image)

        if mask is not None:
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0) # [1 1 h w]
            mask = self.resize_transform(mask)
            mask = (mask > 128).to(dtype=image.dtype, device=image.device)
        inpaint_cond_data = self.mask_processor(image, mask=mask, mask_type_ratio_dict=self.mask_type_ratio_dict_image)
        mask, masked_image = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']   

        image = self.transform(image)
        masked_image = self.transform(masked_image)

        assert image.shape[2] == sample_h, image.shape[3] == sample_w

        image = torch.cat([image, masked_image, mask], dim=1)  #  [1 2C+1 H W]
        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = image_data['cap'] if isinstance(image_data['cap'], list) else [image_data['cap']]
        caps = [random.choice(caps)]
        if '/sam/' in image_data['path']:
            caps = [add_masking_notice(caps[0])]
        if 'ideogram' in image_data['path']:
            caps = [add_high_aesthetic_notice_image(caps[0])]
        if 'civitai' in image_data['path']:
            caps = [add_high_aesthetic_notice_image(caps[0])]
        if 'human_images' in image_data['path']:
            caps = [add_high_aesthetic_notice_image_human(caps[0])]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        text = self.drop(text, is_video=False)['text']

        text_tokens_and_mask_1 = self.tokenizer_1(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_1 = text_tokens_and_mask_1['input_ids']  # 1, l
        cond_mask_1 = text_tokens_and_mask_1['attention_mask']  # 1, l
        
        input_ids_2, cond_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids_2 = text_tokens_and_mask_2['input_ids']  # 1, l
            cond_mask_2 = text_tokens_and_mask_2['attention_mask']  # 1, l

        if self.use_motion:
            motion_score = motion_mapping_fun(image_data['motion_score'])
            return dict(
                pixel_values=image, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, motion_score=motion_score, 
                input_ids_2=input_ids_2, cond_mask_2=cond_mask_2
                )
        else:
            return dict(
                pixel_values=image, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, motion_score=None, 
                input_ids_2=input_ids_2, cond_mask_2=cond_mask_2
                )