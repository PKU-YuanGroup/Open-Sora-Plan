import json
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid

def save_video(video: torch.Tensor):
    import cv2

    frame_count, height, width, channels = video.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (width, height))

    for i in range(frame_count):
        frame = video[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        out.write(frame)


class Inpaint_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.video_data = args.video_data
        self.num_frames = args.num_frames
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.v_decoder = DecordInit()
        self.text_drop_rate = args.cfg

        # inpaint
        # The proportion of executing the i2v task.
        self.i2v_ratio = args.i2v_ratio
        self.transition_ratio = args.transition_ratio
        assert self.i2v_ratio + self.transition_ratio < 1, 'The sum of i2v_ratio and transition_ratio should be less than 1.'
        self.default_text_ratio = args.default_text_ratio

        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
        else:
            raise NotImplementedError('Inpainting dataset only support video data')



    def __len__(self):
        assert self.num_frames != 1, 'Inpainting dataset only support video data'
        return len(self.vid_cap_list)
   
    def __getitem__(self, idx):
        try:
            video_data = {}
            if self.num_frames != 1:
                video_data = self.get_video(idx)
            return dict(video_data=video_data) # video_data: video, input_ids, cond_mask
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_video(self, idx):

        video_path = self.vid_cap_list[idx]['path']
        frame_idx = self.vid_cap_list[idx]['frame_idx']
        video = self.decord_read(video_path, frame_idx)

        video = self.transform(video)

        # video = torch.rand(65, 3, 512, 512)

        video = video.transpose(0, 1)  # T C H W -> C T H W

        # inpaint
        inpaint_cond_data = self.get_mask_masked_video(video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_video']
        video = torch.cat([video, masked_video, mask], dim=0)

        text = self.vid_cap_list[idx]['cap']


        rand_num = random.random()
        if rand_num < self.text_drop_rate:
            text = ''
        elif rand_num < self.text_drop_rate + self.default_text_ratio:
            text = 'The video showcases a scene with coherent and clear visuals.'

    
        text = text_preprocessing(text)
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

    def get_mask_masked_video(self, video):
        mask = torch.zeros_like(video)
        
        rand_num = random.random()
        # To ensure the effectiveness of the i2v task, it is necessary to guarantee that a certain proportion of samples undergo i2v.
        if rand_num < self.i2v_ratio:
            mask = 1 - mask
            mask[:, 0, ...] = 0
        elif rand_num < self.i2v_ratio + self.transition_ratio:
            mask = 1 - mask
            mask[:, 0, ...] = 0
            mask[:, -1, ...] = 0
        else:
            idx_to_select = random.randint(1, self.num_frames - 1)
            selected_indices = random.sample(range(1, self.num_frames), idx_to_select)
            mask[:, selected_indices, ...] = 1

        masked_video = video * (mask < 0.5)
        return dict(mask=mask, masked_video=masked_video)

    def decord_read(self, path, frame_idx=None):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            # start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        
        return video_data


    def get_vid_cap_list(self):
        vid_cap_lists = []
        with open(self.video_data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            new_vid_cap_list = []
            with open(anno, 'r') as f:
                vid_cap_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(vid_cap_list))):
                path = opj(folder, vid_cap_list[i]['path'])
                # For testing, some data has not been utilized.
                if not os.path.exists(path) and not os.path.exists(path.replace('.mp4', '_resize1080p.mp4')):
                    break
                elif os.path.exists(path.replace('.mp4', '_resize1080p.mp4')):
                    path = path.replace('.mp4', '_resize1080p.mp4')
                new_vid_cap_list.append(
                    {
                        'path': path,
                        'frame_idx': vid_cap_list[i]['frame_idx'],
                        'cap': vid_cap_list[i]['cap']
                    }
                )

            vid_cap_lists += new_vid_cap_list
        return vid_cap_lists


if __name__ == "__main__":

    from torchvision import transforms
    from torchvision.transforms import Lambda
    from .transform import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop

    from transformers import AutoTokenizer

    class Args:
        def __init__(self):
            self.video_data = '/storage/gyy/hw/Open-Sora-Plan/scripts/train_data/video_data_debug.txt'
            self.num_frames = 65
            self.model_max_length = 300
            self.cfg = 0.1
            self.i2v_ratio = 0.5
            self.transition_ratio = 0.4
            self.default_text_ratio = 0.1
            self.max_image_size = 512
            self.sample_rate = 1
            self.text_encoder_name = "DeepFloyd/t5-v1_1-xxl"
            self.cache_dir = "/storage/cache_dir"
        
    args = Args()
    resize = [CenterCropResizeVideo((args.max_image_size, args.max_image_size))]

    temporal_sample = TemporalRandomCrop(args.num_frames * args.sample_rate)  # 16 x
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        Lambda(lambda x: 2. * x - 1.)
    ])

    dataset = Inpaint_dataset(args, transform, temporal_sample, tokenizer)

    print(len(dataset))
    data = dataset[0]['video_data']
    video, input_ids, cond_mask = data['video'], data['input_ids'], data['cond_mask']
    print(video.shape) # 3 * C, F, H, W
    print(input_ids.shape) # 1 D
    print(cond_mask.shape) # 1 D


