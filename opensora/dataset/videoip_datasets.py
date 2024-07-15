
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
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
from collections import Counter

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

class VideoIP_dataset(T2V_dataset):
    def __init__(self, args, transform, resize_transform, temporal_sample, tokenizer, transform_topcrop, resize_transform_topcrop, image_processor):
        super().__init__(args, transform, temporal_sample, tokenizer, transform_topcrop)

        self.resize_transform = resize_transform
        self.resize_transform_topcrop = resize_transform_topcrop
        self.image_processor = image_processor

        if self.num_frames != 1:
            # inpaint
            # The proportion of executing the i2v task.
            self.i2v_ratio = args.i2v_ratio
            self.transition_ratio = args.transition_ratio
            self.clear_video_ratio = args.clear_video_ratio
            assert self.i2v_ratio + self.transition_ratio + self.clear_video_ratio < 1, 'The sum of i2v_ratio, transition_ratio and clear video ratio should be less than 1.'
        
        self.default_text_ratio = args.default_text_ratio
        self.default_text = f"The {'video' if self.num_frames != 1 else 'image'} showcases a scene with coherent and clear visuals."

    def get_video(self, idx):

        video_path = dataset_prog.vid_cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        # frame_indice = self.vid_cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path)

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]

        # resize
        video = self.resize_transform(video)

        inpaint_cond_data = self.get_mask_masked_video(video)
        masked_video = inpaint_cond_data['masked_video']

        clip_video = self.image_processor(masked_video) # T C H W

        video = self.transform(video)  # T C H W -> T C H W
        video = video.transpose(0, 1)  # T C H W -> C T H W

        text = dataset_prog.vid_cap_list[idx]['cap']
        if text is None:
            text = self.default_text
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        drop_results = self.drop(text, clip_video)
        text = drop_results['text']
        clip_video = drop_results['clip_image']

        text = text_preprocessing(text, support_Chinese=self.support_Chinese)
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
        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask, clip_video=clip_video)
    
    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames - 1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data['video'][:, i:i + 1] for i in select_image_idx]  # num_img [c, 1, h, w]
        clip_image = [video_data['clip_video'][i:i + 1] for i in select_image_idx]  # num_img [1, c, h, w]
        input_ids = video_data['input_ids'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask, clip_image=clip_image)

    def get_image(self, idx):
        idx = idx % len(dataset_prog.img_cap_list)  # out of range
        image_data = dataset_prog.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        image = [Image.open(i['path']).convert('RGB') for i in image_data]  # num_img [h, w, c]
        image = [torch.from_numpy(np.array(i)) for i in image]  # num_img [h, w, c]

        # for i in image:
        #     assert not torch.any(torch.isnan(i)), 'before transform0'
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image]  # num_img [1 c h w]

        image = [self.resize_transform_topcrop(i) if 'human_images' in j['path'] else self.resize_transform(i) for i, j in zip(image, image_data)]   # num_img [1 C H W] -> num_img [1 C H W]

        clip_image_list = [self.image_processor(i) for i in image] # num_img [1 C H W] -> num_img [1 C H W]
        # for i in image:
        #     assert not torch.any(torch.isnan(i)), 'before transform1'
        # for i in image:
        #     h, w = i.shape[-2:]
        #     assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only image with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But found ratio is {round(h / w, 2)} with the shape of {i.shape}'
        image = [self.transform_topcrop(i) if 'human_images' in j['path'] else self.transform(i) for i, j in zip(image, image_data)]  # num_img [1 C H W] -> num_img [1 C H W]

        # for i in image:
        #     assert not torch.any(torch.isnan(i)), 'after transform'
        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = [i.transpose(0, 1) for i in image]  # num_img [1 C H W] -> num_img [C 1 H W]

        caps = [i['cap'] if isinstance(i['cap'], list) else [i['cap']] for i in image_data]
        caps = [[random.choice(i)] if i is not None or len(i) > 0 else [self.default_text] for i in caps]
        text = [text_preprocessing(cap, support_Chinese=self.support_Chinese) for cap in caps]
        
        input_ids, cond_mask, clip_image = [], [], []
        for t, clip_i in zip(text, clip_image_list):
            drop_results = self.drop(t, clip_i)
            t = drop_results['text']
            clip_i = drop_results['clip_image']
            text_tokens_and_mask = self.tokenizer(
                t,
                max_length=self.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids.append(text_tokens_and_mask['input_ids'])
            cond_mask.append(text_tokens_and_mask['attention_mask'])
            clip_image.append(clip_i)
        input_ids = torch.cat(input_ids)  # self.use_image_num, l
        cond_mask = torch.cat(cond_mask)  # self.use_image_num, l
        clip_image = torch.cat(clip_image) # self.use_image_num, C, H, W
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask, clip_image=clip_image)

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
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.clear_video_ratio:
            pass
        else:
            idx_to_select = random.randint(1, self.num_frames - 1)
            selected_indices = random.sample(range(1, self.num_frames), idx_to_select)
            mask[selected_indices] = 1

        masked_video = video * (mask < 0.5)
        return dict(mask=mask, masked_video=masked_video)
    
    def drop(self, text, clip_image):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''
        elif rand_num < self.cfg * 2:
            clip_image = torch.zeros_like(clip_image, device=clip_image.device, dtype=clip_image.dtype)
        elif rand_num < self.cfg * 3:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''
            clip_image = torch.zeros_like(clip_image, device=clip_image.device, dtype=clip_image.dtype)

        return dict(text=text, clip_image=clip_image)
