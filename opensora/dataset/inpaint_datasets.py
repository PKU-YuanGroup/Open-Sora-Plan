
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

from opensora.models.diffusion.opensora.modeling_inpaint import ModelType, STR_TO_TYPE, TYPE_TO_STR

from .t2v_datasets import filter_json_by_existed_files, random_video_noise, find_closest_y, filter_resolution
from .t2v_datasets import SingletonMeta, DataSetProg
from .t2v_datasets import T2V_dataset

import imageio

logger = get_logger(__name__)

dataset_prog = DataSetProg()

def save_video(video, name='video.mp4'):
    imageio.mimwrite(
        name, video, fps=24, quality=6)  # highest quality is 10, lowest is 0

def get_inpaint_dataset(model_type):
    model_type = STR_TO_TYPE[model_type]
    if model_type == ModelType.INPAINT_ONLY:
        return Inpaint_dataset
    elif model_type == ModelType.VIP_ONLY:
        return VIP_dataset
    elif model_type == ModelType.VIP_INPAINT:
        return VIPInpaint_dataset
    else:
        raise NotImplementedError(f"Model type {TYPE_TO_STR[model_type]} not implemented.")


class Meta_dataset(T2V_dataset):
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
            self.v2v_ratio = args.v2v_ratio
            self.clear_video_ratio = args.clear_video_ratio
            assert self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio < 1, 'The sum of i2v_ratio, transition_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
        self.default_text_ratio = args.default_text_ratio
        self.default_text = f"The {'video' if self.num_frames != 1 else 'image'} showcases a scene with coherent and clear visuals."
    
    def get_mask_masked_video(self, video):
        # video shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = video.shape
        mask = torch.ones_like(video, device=video.device, dtype=video.dtype)
        
        rand_num = random.random()
        # i2v
        if rand_num < self.i2v_ratio:
            mask[0] = 0
        # transition
        elif rand_num < self.i2v_ratio + self.transition_ratio:
            mask[0] = 0
            mask[-1] = 0
        # video continuation
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio:
            end_idx = random.randint(1, t)
            mask[:end_idx] = 0
        # clear video
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio:
            mask[:] = 0
        # random mask
        else:
            idx_to_select = random.randint(0, t - 1)
            selected_indices = random.sample(range(0, t), idx_to_select)
            mask[selected_indices] = 0
        masked_video = video * (mask < 0.5)

        # save_video(masked_video.permute(0, 2, 3, 1).cpu().numpy(), 'masked_video.mp4')
        return dict(mask=mask, masked_video=masked_video)

    def drop(self, text):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''

        return dict(text=text)

    
class Inpaint_dataset(Meta_dataset):
    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        # frame_indice = self.cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path)

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]

        video = self.resize_transform(video)
        video = self.transform(video)  # T C H W -> T C H W

        # inpaint
        inpaint_cond_data = self.get_mask_masked_video(video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_video']
        video = torch.cat([video, masked_video, mask], dim=1) # T 3*C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.cap_list[idx]['cap']
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
        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask)
    
    
class VIP_dataset(Meta_dataset):
    def __init__(self, args, transform, resize_transform, temporal_sample, tokenizer, transform_topcrop, resize_transform_topcrop, image_processor):
        super().__init__(args, transform, resize_transform, temporal_sample, tokenizer, transform_topcrop, resize_transform_topcrop, image_processor)
        self.use_clip_mask = args.use_clip_mask

    def drop(self, text, clip_image, clip_mask=None):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''
        elif rand_num < self.cfg * 2:
            clip_image = torch.zeros_like(clip_image, device=clip_image.device, dtype=clip_image.dtype)
            clip_mask = torch.ones_like(clip_mask, device=clip_mask.device, dtype=clip_mask.dtype) if clip_mask is not None else None
        elif rand_num < self.cfg * 3:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''
            clip_image = torch.zeros_like(clip_image, device=clip_image.device, dtype=clip_image.dtype)
            clip_mask = torch.ones_like(clip_mask, device=clip_mask.device, dtype=clip_mask.dtype) if clip_mask is not None else None

        return dict(text=text, clip_image=clip_image, clip_mask=clip_mask)
    

    def get_mask_masked_video(self, video):
        # video shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = video.shape
        mask = torch.ones_like(video, device=video.device, dtype=video.dtype)
        clip_mask = torch.ones([video.shape[0], 1, 1, 1]) if self.use_clip_mask else None
        
        rand_num = random.random()
        # i2v
        if rand_num < self.i2v_ratio:
            mask[0] = 0
            if self.use_clip_mask:
                clip_mask[0] = 0
        # transition
        elif rand_num < self.i2v_ratio + self.transition_ratio:
            mask[0] = 0
            mask[-1] = 0
            if self.use_clip_mask:
                clip_mask[0] = 0
                clip_mask[-1] = 0
        # video continuation
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio:
            end_idx = random.randint(1, t)
            mask[:end_idx] = 0
            if self.use_clip_mask:
                clip_mask[:end_idx] = 0
        # clear video
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio:
            mask[:] = 0
            if self.use_clip_mask:
                clip_mask[:] = 0
        # random mask
        else:
            idx_to_select = random.randint(0, t - 1)
            selected_indices = random.sample(range(0, t), idx_to_select)
            mask[selected_indices] = 0
            if self.use_clip_mask:
                clip_mask[selected_indices] = 0

        masked_video = video * (mask < 0.5)
        return dict(mask=mask, masked_video=masked_video, clip_mask=clip_mask)


    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        # import ipdb;ipdb.set_trace()
        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]
        # for i in image:
        #     h, w = i.shape[-2:]
        #     assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only image with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But found ratio is {round(h / w, 2)} with the shape of {i.shape}'
        
        image = self.resize_transform_topcrop(image) if 'human_images' in image_data['path'] else self.resize_transform(image)  #  [1 C H W] -> [1 C H W]

        clip_image = self.image_processor(image) # [1 C H W]

        image = self.transform_topcrop(image) if 'human_images' in image_data['path'] else self.transform(image) #  [1 C H W] -> [1 C H W]

        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = image_data['cap'] if isinstance(image_data['cap'], list) else [image_data['cap']]
        caps = [random.choice(caps)]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        
        drop_results = self.drop(text, clip_image)
        text = drop_results['text']
        clip_image = drop_results['clip_image']

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

        return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask, clip_data=clip_image, clip_mask=None)

    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        # frame_indice = self.cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path)

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]

        video = self.resize_transform(video)

        inpaint_cond_data = self.get_mask_masked_video(video)
        masked_video = inpaint_cond_data['masked_video']
        clip_video = self.image_processor(masked_video) # T C H W

        clip_mask = inpaint_cond_data['clip_mask'] # T 1 1 1 

        video = self.transform(video)  # T C H W -> T C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.cap_list[idx]['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text_preprocessing(text, support_Chinese=self.support_Chinese)

        drop_results = self.drop(text, clip_video, clip_mask=clip_mask)
        text = drop_results['text']
        clip_video = drop_results['clip_image']
        clip_mask = drop_results['clip_mask']

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
        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask, clip_data=clip_video, clip_mask=clip_mask)


class VIPInpaint_dataset(VIP_dataset):
    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        # frame_indice = self.cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path)

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]

        video = self.resize_transform(video)

        inpaint_cond_data = self.get_mask_masked_video(video)
        masked_video = inpaint_cond_data['masked_video']
        mask = inpaint_cond_data['mask']
        clip_video = self.image_processor(masked_video) # T C H W

        clip_mask = inpaint_cond_data['clip_mask'] # T 1 1 1

        video = torch.cat([video, masked_video], dim=1) # T 2*C H W
        video = self.transform(video)  # T C H W -> T C H W

        video = torch.cat([video, mask], dim=1) # T 3*C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.cap_list[idx]['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text_preprocessing(text, support_Chinese=self.support_Chinese)

        drop_results = self.drop(text, clip_video, clip_mask=clip_mask)
        text = drop_results['text']
        clip_video = drop_results['clip_image']
        clip_mask = drop_results['clip_mask']

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
        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask, clip_data=clip_video, clip_mask=clip_mask)
    
if __name__ == "__main__":

    from .transform import CenterCropResizeVideo, ToTensorAfterResize, RandomHorizontalFlipVideo, NormalizeVideo, TemporalRandomCrop
    from transformers import AutoTokenizer
    from torchvision.transforms import Lambda

    class Args:
        max_height = 480
        max_width = 640
        model_max_length = 512
        cache_dir = '../cache_dir'
        data = '/storage/gyy/hw/Open-Sora-Plan/scripts/train_data/video_data.txt'
        num_frames = 93
        train_fps = 24
        use_image_num = 0
        use_img_from_vid = False
        cfg = 0.05
        speed_factor = 1.0
        drop_short_ratio = 0.0
        text_encoder_name = 'google/mt5-xxl'
        dataloader_num_workers = 10
        i2v_ratio = 0.0
        transition_ratio = 0.0
        v2v_ratio = 0.0
        clear_video_ratio = 0.9
        default_text_ratio = 0.1

    args = Args()

    temporal_sample = TemporalRandomCrop(args.num_frames)
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    resize_topcrop = CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True)
    resize = CenterCropResizeVideo((args.max_height, args.max_width))
    transform = transforms.Compose([
        ToTensorAfterResize(),
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])
    transform_topcrop = transforms.Compose([
        ToTensorAfterResize(),
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])

    # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)

    dataset = Inpaint_dataset(args, transform=transform, resize_transform=resize, resize_transform_topcrop=resize_topcrop, temporal_sample=temporal_sample, tokenizer=tokenizer, transform_topcrop=transform_topcrop, image_processor=None)
    print(len(dataset))
    results = dataset[1]
    video = results['pixel_values']
    input_ids = results['input_ids']
    cond_mask = results['cond_mask']
    print(video.shape, input_ids.shape, cond_mask.shape)
