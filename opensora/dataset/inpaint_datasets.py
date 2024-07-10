<<<<<<< HEAD
import json
import os, io, csv, math, random
from turtle import width
=======

from torch.utils.data import Dataset

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import glob
import json
import os, io, csv, math, random
>>>>>>> new_hw_onging
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
<<<<<<< HEAD
=======
from collections import Counter
>>>>>>> new_hw_onging

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing

from .t2v_datasets import filter_json_by_existed_files, random_video_noise, find_closest_y, filter_resolution
from .t2v_datasets import SingletonMeta, DataSetProg
from .t2v_datasets import T2V_dataset

logger = get_logger(__name__)


dataset_prog = DataSetProg()

class Inpaint_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        super().__init__(args, transform, temporal_sample, tokenizer, transform_topcrop)

        # inpaint
        # The proportion of executing the i2v task.
        self.i2v_ratio = args.i2v_ratio
        self.transition_ratio = args.transition_ratio
        assert self.i2v_ratio + self.transition_ratio < 1, 'The sum of i2v_ratio and transition_ratio should be less than 1.'
        self.default_text_ratio = args.default_text_ratio

    def get_video(self, idx):
        # npu_config.print_msg(f"current idx is {idx}")
        # video = random.choice([random_video_noise(65, 3, 336, 448), random_video_noise(65, 3, 1024, 1024), random_video_noise(65, 3, 360, 480)])
        # # print('random shape', video.shape)
        # input_ids = torch.ones(1, 120).to(torch.long).squeeze(0)
        # cond_mask = torch.cat([torch.ones(1, 60).to(torch.long), torch.ones(1, 60).to(torch.long)], dim=1).squeeze(0)

        video_path = dataset_prog.vid_cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        # frame_indice = self.vid_cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path)

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]
        video = self.transform(video)  # T C H W -> T C H W

        # inpaint
        inpaint_cond_data = self.get_mask_masked_video(video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_video']
        video = torch.cat([video, masked_video, mask], dim=1) # T 3*C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.vid_cap_list[idx]['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text_preprocessing(text, support_Chinese=self.support_Chinese)
        
        text = self.drop(text)['text']

        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)
    
    def drop(self, text):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = 'The video showcases a scene with coherent and clear visuals.' if rand_num_text < self.default_text_ratio else ''

        return dict(text=text)

    def get_mask_masked_video(self, video):
        # video shape (T, C, H, W)
        mask = torch.zeros_like(video)
        
        rand_num = random.random()
        # To ensure the effectiveness of the i2v task, it is necessary to guarantee that a certain proportion of samples undergo i2v.
        if rand_num < self.i2v_ratio:
            mask = 1 - mask
            mask[0] = 0
        elif rand_num < self.i2v_ratio + self.transition_ratio:
            mask = 1 - mask
            mask[0] = 0
            mask[-1] = 0
        else:
            idx_to_select = random.randint(1, self.num_frames - 1)
            selected_indices = random.sample(range(1, self.num_frames), idx_to_select)
            mask[selected_indices] = 1

        masked_video = video * (mask < 0.5)
        return dict(mask=mask, masked_video=masked_video)
