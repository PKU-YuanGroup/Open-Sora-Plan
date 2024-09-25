
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

from opensora.dataset.inpaint_utils import get_mask_tensor,MaskType
from ultralytics import YOLO

logger = get_logger(__name__)

dataset_prog = DataSetProg()

def save_video(video, name='video.mp4'):
    imageio.mimwrite(
        name, video, fps=24, quality=6)  # highest quality is 10, lowest is 0

class Meta_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        super().__init__(args, transform, temporal_sample, tokenizer)

    #     if self.num_frames != 1:
    #         # inpaint
    #         self.t2v_ratio = args.t2v_ratio
    #         self.i2v_ratio = args.i2v_ratio
    #         self.transition_ratio = args.transition_ratio
    #         self.v2v_ratio = args.v2v_ratio
    #         self.clear_video_ratio = args.clear_video_ratio
    #         self.Semantic_ratio = args.Semantic_ratio
    #         self.bbox_ratio = args.bbox_ratio
    #         self.background_ratio = args.background_ratio
    #         self.fixed_ratio = args.fixed_ratio
    #         self.Semantic_expansion_ratio = args.Semantic_expansion_ratio
    #         self.fixed_bg_ratio = args.fixed_bg_ratio
    #         assert self.t2v_ratio + self.i2v_ratio + self.transition_ratio + self.v2v_ratio + self.clear_video_ratio + self.Semantic_ratio + self.bbox_ratio + self.background_ratio + self.fixed_ratio + self.fixed_bg_ratio + self.Semantic_expansion_ratio < 1, 'The sum of t2v_ratio, i2v_ratio, transition_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
    #     self.min_clear_ratio = 0.0 if args.min_clear_ratio is None else args.min_clear_ratio
    #     assert self.min_clear_ratio >= 0 and self.min_clear_ratio <= 1, 'min_clear_ratio should be in the range of [0, 1].'

    #     self.mask_processor = mask_processor
    #     self.init_mask_func()

        self.default_text_ratio = args.default_text_ratio

    #     self.yolomodel = modelYOLO
    #     # self.yolomodel = None

    # def init_mask_func(self):
    #     # mask: ones_like (t 1 h w)
    #     def t2iv(mask):
    #         mask[:] = 1
    #         return mask
        
    #     def i2v(mask):
    #         mask[0] = 0
    #         return mask
        
    #     def transition(mask):
    #         mask[0] = 0
    #         mask[-1] = 0
    #         return mask
        
    #     def v2v(mask):
    #         end_idx = random.randint(int(mask.shape[0] * self.min_clear_ratio), mask.shape[0])
    #         mask[:end_idx] = 0
    #         return mask
        
    #     def clear(mask):
    #         mask[:] = 0
    #         return mask
        
    #     def random_mask(mask):
    #         num_to_select = random.randint(int(mask.shape[0] * self.min_clear_ratio), mask.shape[0])
    #         selected_indices = random.sample(range(mask.shape[0]), num_to_select)
    #         mask[selected_indices] = 0
    #         return mask
        
    #     def Semantic_mask(video_tensor):
    #         return get_mask_tensor(video_tensor,MaskType.Semantic_mask,self.yolomodel)

    #     def bbox_mask(video_tensor):
    #         return get_mask_tensor(video_tensor,MaskType.bbox_mask,self.yolomodel)
        
    #     def background_mask(video_tensor):
    #         return get_mask_tensor(video_tensor,MaskType.background_mask,self.yolomodel)
        
    #     def fixed_mask(video_tensor):
    #         return get_mask_tensor(video_tensor,MaskType.fixed_mask,self.yolomodel)

    #     def Semantic_expansion_mask(video_tensor):
    #         return get_mask_tensor(video_tensor,MaskType.Semantic_expansion_mask,self.yolomodel)
        
    #     def fixed_bg_mask(video_tensor):
    #         return get_mask_tensor(video_tensor,MaskType.fixed_bg_mask,self.yolomodel)



    #     self.mask_functions = {
    #         't2iv': t2iv,
    #         'i2v': i2v,
    #         'transition': transition,
    #         'v2v': v2v,
    #         'clear': clear,
    #         'random_mask': random_mask,
    #         'Semantic_mask':Semantic_mask,
    #         'bbox_mask':bbox_mask,
    #         'background_mask':background_mask,
    #         'fixed_mask':fixed_mask,
    #         'Semantic_expansion_mask':Semantic_expansion_mask,
    #         'fixed_bg_mask':fixed_bg_mask
    #     }
        
    
    # def get_mask_masked_pixel_values(self, pixel_values, mask_func_weights):
    #     # pixel_values shape (T, C, H, W)
    #     # 1 means masked, 0 means not masked
    #     t, c, h, w = pixel_values.shape
    #     mask = torch.ones([t, 1, h, w], device=pixel_values.device, dtype=pixel_values.dtype)
        
    #     mask_func_name = random.choices(list(mask_func_weights.keys()), list(mask_func_weights.values()))[0]
    #     frame_mask_list = ['t2iv','i2v','transition','v2v','clear','random_mask']
    #     pos_mask_list = ['Semantic_mask','bbox_mask','background_mask','fixed_mask','Semantic_expansion_mask','fixed_bg_mask']

    #     if mask_func_name in frame_mask_list:
    #         mask = self.mask_functions[mask_func_name](mask)
    #         masked_pixel_values = pixel_values * (mask < 0.5)
        
    #     if mask_func_name in pos_mask_list:
    #         masked_pixel_values,mask = self.mask_functions[mask_func_name](pixel_values)
    #     # save_video(masked_pixel_values.permute(0, 2, 3, 1).cpu().numpy(), 'masked_video.mp4')
    #     return dict(mask=mask, masked_pixel_values=masked_pixel_values)

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

    
class Inpaint_dataset(Meta_dataset):

    def __init__(self, args, transform, temporal_sample, tokenizer):
        super().__init__(args, transform, temporal_sample, tokenizer)

        # self.mask_func_weights_video = {
        #     't2iv': self.t2v_ratio, 
        #     'i2v': self.i2v_ratio, 
        #     'transition': self.transition_ratio, 
        #     'v2v': self.v2v_ratio, 
        #     'clear': self.clear_video_ratio, 
        #     'Semantic_mask':self.Semantic_ratio,
        #     'bbox_mask':self.bbox_ratio,
        #     'background_mask':self.background_ratio,
        #     'fixed_mask':self.fixed_ratio,
        #     'Semantic_expansion_mask':self.Semantic_expansion_ratio,
        #     'fixed_bg_mask':self.fixed_bg_ratio,
        #     'random_mask': 1 - self.t2v_ratio - self.i2v_ratio - self.transition_ratio - self.v2v_ratio - self.clear_video_ratio - self.Semantic_ratio - self.bbox_ratio - self.background_ratio - self.fixed_ratio - self.Semantic_expansion_ratio - self.fixed_bg_ratio
            
        # }

        # self.mask_func_weights_image = {
        #     't2iv': 0.9, 
        #     'clear': 0.1
        # }

    def get_video(self, idx):
        video_data = dataset_prog.cap_list[idx]
        video_path = video_data['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']
        if self.video_reader == 'decord':
            video = self.decord_read(video_path, predefine_frame_indice=frame_indice)
        elif self.video_reader == 'opencv':
            video = self.opencv_read(video_path, predefine_frame_indice=frame_indice)
        else:
            NotImplementedError(f'Found {self.video_reader}, but support decord or opencv')

        # inpaint_cond_data = self.get_mask_masked_pixel_values(video, self.mask_func_weights_video)
        # mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        video = self.transform(video)  # T C H W -> T C H W
        # masked_video = self.transform(masked_video)  # T C H W -> T C H W
        # mask = self.mask_processor(mask)  # T 1 H W -> T 1 H W
        assert video.shape[2] == sample_h and video.shape[3] == sample_w

        # video = torch.cat([video, masked_video, mask], dim=1)  # T 2C+1 H W
        # video = torch.rand(221, 3, 480, 640)

        # video = video.transpose(0, 1)  # T C H W -> C T H W
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

        # inpaint_cond_data = self.get_mask_masked_pixel_values(image, self.mask_func_weights_image)
        # mask, masked_image = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        # import ipdb;ipdb.set_trace()
        image = self.transform(image) #  [1 C H W] -> [1 C H W]
        # masked_image = self.transform(masked_image) #  [1 C H W] -> [1 C H W]
        # mask = self.mask_processor(mask) #  [1 1 H W] -> [1 1 H W]
        assert image.shape[2] == sample_h, image.shape[3] == sample_w

        # image = torch.cat([image, masked_image, mask], dim=1)  #  [1 2C+1 H W]

        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        # image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

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
        text = self.drop(text, is_video=False)['text']

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
    

# class AllInpaintDataset(Inpaint_dataset):
    

if __name__ == "__main__":
    class Args:
        t2v_ratio = 0.0
        i2v_ratio = 0.0
        transition_ratio = 0.0
        v2v_ratio = 0.00
        clear_video_ratio = 0.99
        min_clear_ratio = 0.0
        Semantic_ratio = 0.0
        bbox_ratio = 0.0
        background_ratio = 0.0
        fixed_ratio = 0.0
        Semantic_expansion_ratio = 0.0
        fixed_bg_ratio = 0.0
        default_text_ratio = 0.1
        use_motion = False
        support_Chinese = False
        model_max_length = 512
        cfg = 0.1
        num_frames = 93
        force_resolution = False
        max_height = 320
        max_width = 320
        hw_stride = 32
        data = "/storage/gyy/hw/Open-Sora-Plan/scripts/train_data/merge_data_debug.txt"
        train_fps = 16
        use_image_num = 0
        use_img_from_vid = False
        speed_factor = 1.0
        drop_short_ratio = 0.0
        cfg = 0.1
        dataloader_num_workers = 4
        use_motion = True
        skip_low_resolution = True
        text_encoder_name = 'google/mt5-xxl'


    from transformers import AutoTokenizer

    from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize
    from torchvision.transforms import Lambda

    args = Args()
    
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    if args.force_resolution:
        resize = [CenterCropResizeVideo((args.max_height, args.max_width)), ]
    else:
        resize = [
            LongSideResizeVideo((args.max_height, args.max_width), skip_low_resolution=True), 
            SpatialStrideCropVideo(stride=args.hw_stride), 
        ]
    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        norm_fun
    ])
    tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl")

    mask_processor = transforms.Compose([*resize])

    dataset = Inpaint_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer, mask_processor=mask_processor)

    data = next(iter(dataset))

    # print(data['pixel_values'].shape)
    # print(data['input_ids'].shape)
    # print(data['cond_mask'].shape)
    # print(data['motion_score'])

    # print(data['pixel_values'])
    # print(data['input_ids'])
    # print(data['cond_mask'])
