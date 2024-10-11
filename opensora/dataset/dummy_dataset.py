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
from opensora.dataset.transform import get_params, maxhwresize, add_masking_notice, calculate_statistics, \
    add_aesthetic_notice_image, add_aesthetic_notice_video

import decord
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = get_logger(__name__)


def filter_json_by_existed_files(directory, data, postfix=".mp4"):
    # 构建搜索模式，以匹配指定后缀的文件
    pattern = os.path.join(directory, '**', f'*{postfix}')
    mp4_files = glob.glob(pattern, recursive=True)  # 使用glob查找所有匹配的文件

    # 使用文件的绝对路径构建集合
    mp4_files_set = set(os.path.abspath(path) for path in mp4_files)

    # 过滤数据条目，只保留路径在mp4文件集合中的条目
    filtered_items = [item for item in data if item['path'] in mp4_files_set]

    return filtered_items


def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid


class Dummy_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer_1, tokenizer_2):
        self.data = args.data
        self.num_frames = args.num_frames
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.support_Chinese = False
        if 'mt5' in args.text_encoder_name_1:
            self.support_Chinese = True
        if args.text_encoder_name_2 is not None and 'mt5' in args.text_encoder_name_2:
            self.support_Chinese = True

        self.num_test_samples = args.num_test_samples
        self.image_data_ratio = args.image_data_ratio


    def __len__(self):
        return self.num_test_samples

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_data(self, idx):
        if random.random() < self.image_data_ratio:
            return self.get_image(idx)
        else:
            return self.get_video(idx)
    
    def get_video(self, idx):

        video = random_video_noise(self.num_frames, 3, self.max_height, self.max_width)
        # import ipdb;ipdb.set_trace()
        video = self.transform(video)  # T C H W -> T C H W

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = ['dummy text']
        text = [random.choice(text)]

        text = text if random.random() > self.cfg else ""

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

        return dict(
            pixel_values=video, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, 
            input_ids_2=input_ids_2, cond_mask_2=cond_mask_2,
            )

    def get_image(self, idx):
        image = random_video_noise(1, 3, self.max_height, self.max_width)

        image = self.transform(image) #  [1 C H W] -> num_img [1 C H W]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = ['dummy text']
        caps = [random.choice(caps)]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        text = text if random.random() > self.cfg else ""

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

        return dict(
            pixel_values=image, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, 
            input_ids_2=input_ids_2, cond_mask_2=cond_mask_2
            )
