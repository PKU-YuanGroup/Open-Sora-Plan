
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
from opensora.dataset.transform import add_masking_notice, motion_mapping_fun, add_webvid_watermark_notice, \
    clean_vidal, add_high_aesthetic_notice_image, add_aesthetic_notice_video, add_high_aesthetic_notice_image_human

from opensora.dataset.t2v_datasets import SingletonMeta, DataSetProg
from opensora.dataset.t2v_datasets import T2V_dataset

import imageio

logger = get_logger(__name__)

dataset_prog = DataSetProg()

def save_video(video, name='video.mp4'):
    imageio.mimwrite(
        name, video, fps=24, quality=6)  # highest quality is 10, lowest is 0

class Meta_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        super().__init__(args, transform, temporal_sample, tokenizer)

        if self.num_frames != 1:
            # inpaint
            self.t2v_ratio = args.t2v_ratio
            self.i2v_ratio = args.i2v_ratio
            self.transition_ratio = args.transition_ratio
            self.v2v_ratio = args.v2v_ratio
            self.clear_video_ratio = args.clear_video_ratio
            assert self.t2v_ratio + self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio < 1, 'The sum of t2v_ratio, i2v_ratio, transition_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
        self.min_mask_ratio = 0.0 if args.min_mask_ratio is None else args.min_mask_ratio
        assert self.min_mask_ratio >= 0 and self.min_mask_ratio <= 1, 'min_mask_ratio should be in the range of [0, 1].'

        self.init_mask_func()

        self.default_text_ratio = args.default_text_ratio
        self.default_text = f"The {'video' if self.num_frames != 1 else 'image'} showcases a scene with coherent and clear visuals."

    def init_mask_func(self):
        def t2v(mask):
            mask[:] = 1
            return mask
        
        def i2v(mask):
            mask[0] = 0
            return mask
        
        def transition(mask):
            mask[0] = 0
            mask[-1] = 0
            return mask
        
        def v2v(mask):
            end_idx = random.randint(int(self.min_mask_ratio * mask.shape[0]), mask.shape[0])
            mask[:end_idx] = 0
            return mask
        
        def clear(mask):
            mask[:] = 0
            return mask
        
        def random_mask(mask):
            idx_to_select = random.randint(int(self.min_mask_ratio * mask.shape[0]), mask.shape[0])
            selected_indices = random.sample(range(mask.shape[0]), idx_to_select)
            mask[selected_indices] = 0
            return mask
        
        self.mask_functions = {
            't2v': t2v,
            'i2v': i2v,
            'transition': transition,
            'v2v': v2v,
            'clear': clear,
            'random_mask': random_mask
        }
        
    
    def get_mask_masked_pixel_values(self, pixel_values, mask_func_weights):
        # pixel_values shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = pixel_values.shape
        mask = torch.ones([t, 1, h, w], device=pixel_values.device, dtype=pixel_values.dtype)
        
        mask_func_name = random.choices(list(mask_func_weights.keys()), list(mask_func_weights.values()))[0]
        mask = self.mask_functions[mask_func_name](mask)

        masked_pixel_values = pixel_values * (mask < 0.5)
        # save_video(masked_pixel_values.permute(0, 2, 3, 1).cpu().numpy(), 'masked_video.mp4')
        return dict(mask=mask, masked_pixel_values=masked_pixel_values)

    def drop(self, text):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''

        return dict(text=text)

    
class Inpaint_dataset(Meta_dataset):

    def __init__(self, args, transform, temporal_sample, tokenizer):
        super().__init__(args, transform, temporal_sample, tokenizer)

        self.mask_func_weights_video = {
            't2v': self.t2v_ratio, 
            'i2v': self.i2v_ratio, 
            'transition': self.transition_ratio, 
            'v2v': self.v2v_ratio, 
            'clear': self.clear_video_ratio, 
            'random_mask': 1 - self.t2v_ratio - self.i2v_ratio - self.transition_ratio - self.v2v_ratio - self.clear_video_ratio
        }

        self.mask_func_weights_image = {
            't2v': 0.9, 
            'clear': 0.1
        }

    def get_video(self, idx):
        video_data = dataset_prog.cap_list[idx]
        video_path = video_data['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']
        video = self.decord_read(video_path, predefine_frame_indice=frame_indice)
        # import ipdb;ipdb.set_trace()

        inpaint_cond_data = self.get_mask_masked_pixel_values(video, self.mask_func_weights_video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        video = self.transform(video)  # T C H W -> T C H W
        masked_video = self.transform(masked_video)  # T C H W -> T C H W
        assert video.shape[2] == sample_h and video.shape[3] == sample_w

        video = torch.cat([video, masked_video, mask], dim=1)  # T 2C+1 H W
        # video = torch.rand(221, 3, 480, 640)

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
        if self.use_motion:
            motion_score = motion_mapping_fun(video_data['motion_score'])
            return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask, motion_score=motion_score)
        else:
            return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask, motion_score=None)
        
    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        sample_h = image_data['resolution']['sample_height']
        sample_w = image_data['resolution']['sample_width']

        # import ipdb;ipdb.set_trace()
        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]

        inpaint_cond_data = self.get_mask_masked_pixel_values(image, self.mask_func_weights_image)
        mask, masked_image = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        # import ipdb;ipdb.set_trace()
        image = self.transform(image) #  [1 C H W] -> num_img [1 C H W]
        masked_image = self.transform(masked_image) #  [1 C H W] -> num_img [1 C H W]
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
        if 'human_images' in image_data['path']:
            caps = [add_high_aesthetic_notice_image_human(caps[0])]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        input_ids, cond_mask = [], []
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
        input_ids = text_tokens_and_mask['input_ids']  # 1, l
        cond_mask = text_tokens_and_mask['attention_mask']  # 1, l
        if self.use_motion:
            motion_score = motion_mapping_fun(image_data['motion_score'])
            return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask, motion_score=motion_score)
        else:
            return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask, motion_score=None)
    