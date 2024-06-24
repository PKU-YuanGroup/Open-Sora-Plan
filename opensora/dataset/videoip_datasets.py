import time
import traceback


import glob
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
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


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


class VideoIP_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.image_data = args.image_data
        self.video_data = args.video_data
        self.num_frames = args.num_frames
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.v_decoder = DecordInit()

        self.support_Chinese = True
        if not ('mt5' in args.text_encoder_name):
            self.support_Chinese = False

        if self.num_frames != 1:
            self.vid_cap_list = self.get_vid_cap_list()
            if self.use_image_num != 0 and not self.use_img_from_vid:
                self.img_cap_list = self.get_img_cap_list()
            else:
                self.img_cap_list = []
        else:
            self.img_cap_list = self.get_img_cap_list()
            self.vid_cap_list = []

        # inpaint
        # The proportion of executing the i2v task.
        self.i2v_ratio = args.i2v_ratio
        self.transition_ratio = args.transition_ratio
        self.clear_video_ratio = args.clear_video_ratio
        assert self.i2v_ratio + self.transition_ratio + self.clear_video_ratio < 1, 'The sum of i2v_ratio, transition_ratio and clear video ratio should be less than 1.'

        print(f"video length: {len(self.vid_cap_list)}")
        print(f"image length: {len(self.img_cap_list)}")

    def set_checkpoint(self, n_used_elements):
        self.n_used_elements = n_used_elements

    def __len__(self):
        if self.num_frames != 1:
            return len(self.vid_cap_list)
        else:
            return len(self.img_cap_list)

    def __getitem__(self, idx):

        video_data, image_data = {}, {}
        try:
            if self.num_frames != 1:
                video_data = self.get_video(idx)
                if self.use_image_num != 0:
                    if self.use_img_from_vid:
                        image_data = self.get_image_from_video(video_data)
                    else:
                        image_data = self.get_image(idx)
            else:
                image_data = self.get_image(idx)  # 1 frame video as image
            return dict(video_data=video_data, image_data=image_data)
        except Exception as e:
            # print(f'Error with {e}')
            # 打印异常堆栈
            if idx in self.vid_cap_list:
                print(f"Caught an exception! {self.vid_cap_list[idx]}")
            # traceback.print_exc()
            # traceback.print_stack()
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_video(self, idx):

        video_path = self.vid_cap_list[idx]['path']
        assert os.path.exists(video_path) and os.path.getsize(video_path) > 10240, f"file {video_path} has wrong size!"
        frame_idx = self.vid_cap_list[idx]['frame_idx'] if hasattr(self.vid_cap_list[idx], 'frame_idx') else None
        video = self.decord_read(video_path, frame_idx)

        video = self.transform(video)  # T C H W -> T C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = self.vid_cap_list[idx]['cap']

        # inpaint
        inpaint_cond_data = self.get_mask_masked_video(video)
        masked_video = inpaint_cond_data['masked_video']
        video = torch.cat([video, masked_video], dim=0)

        text = text_preprocessing(text, support_Chinese=self.support_Chinese) if random.random() > self.cfg else ""
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

    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames - 1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data['video'][:, i:i + 1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data['input_ids'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

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
        elif rand_num < self.i2v_ratio + self.transition_ratio + self.clear_video_ratio:
            pass
        else:
            idx_to_select = random.randint(1, self.num_frames - 1)
            selected_indices = random.sample(range(1, self.num_frames), idx_to_select)
            mask[:, selected_indices, ...] = 1

        masked_video = video * (mask < 0.5)
        return dict(mask=mask, masked_video=masked_video)

    def get_image(self, idx):
        idx = idx % len(self.img_cap_list)  # out of range
        image_data = self.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        image = [Image.open(i['path']).convert('RGB') for i in image_data]  # num_img [h, w, c]
        image = [torch.from_numpy(np.array(i)) for i in image]  # num_img [h, w, c]

        for i in image:
            assert not torch.any(torch.isnan(i)), 'before transform0'
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image]  # num_img [1 c h w]
        for i in image:
            assert not torch.any(torch.isnan(i)), 'before transform1'
        image = [self.transform(i) for i in image]  # num_img [1 C H W] -> num_img [1 C H W]

        for i in image:
            assert not torch.any(torch.isnan(i)), 'after transform'
        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = [i.transpose(0, 1) for i in image]  # num_img [1 C H W] -> num_img [C 1 H W]

        caps = [i['cap'] for i in image_data]
        text = [text_preprocessing(cap, support_Chinese=self.support_Chinese) for cap in caps]
        input_ids, cond_mask = [], []
        for t in text:
            t = t if random.random() > self.cfg else ""
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
        input_ids = torch.cat(input_ids)  # self.use_image_num, l
        cond_mask = torch.cat(cond_mask)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

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

    def read_jsons(self, data, postfix=".jpg"):
        cap_lists = []
        with open(data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                sub_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(sub_list))):
                sub_list[i]['path'] = opj(folder, sub_list[i]['path'])
            cap_lists += sub_list
        return cap_lists

    def get_img_cap_list(self):
        use_image_num = self.use_image_num if self.use_image_num != 0 else 1
        img_cap_lists = self.read_jsons(self.image_data, postfix=".jpg")
        img_cap_lists = [img_cap_lists[i: i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
       
        return img_cap_lists[:-1]  # drop last to avoid error length

    def get_vid_cap_list(self):
        vid_cap_lists = self.read_jsons(self.video_data, postfix=".mp4")

        return vid_cap_lists


if __name__ == "__main__":

    from torchvision import transforms
    from torchvision.transforms import Lambda
    from .transform import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop

    from transformers import AutoTokenizer

    class Args:
        def __init__(self):
            self.video_data = '/remote-home/gyy/Open-Sora-Plan/scripts/train_data/video_data_debug.txt'
            self.image_data = '/remote-home/gyy/Open-Sora-Plan/scripts/train_data/image_data_debug.txt'
            self.num_frames = 65
            self.use_image_num = 4
            self.use_img_from_vid = False
            self.model_max_length = 300
            self.cfg = 0.1
            self.i2v_ratio = 0.3
            self.transition_ratio = 0.3
            self.clear_video_ratio = 0.3
            self.max_image_size = 512
            self.sample_rate = 1
            self.text_encoder_name = "DeepFloyd/t5-v1_1-xxl"
            self.cache_dir = "./cache_dir"
        
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

    dataset = VideoIP_dataset(args, transform, temporal_sample, tokenizer)

    print(len(dataset))
    video_data = dataset[0]['video_data']
    image_data = dataset[0]['image_data']
    video, video_input_ids, video_cond_mask = video_data['video'], video_data['input_ids'], video_data['cond_mask']
    image, image_input_ids, image_cond_mask = image_data['image'], image_data['input_ids'], image_data['cond_mask']
    print(video.shape) # 3 * C, F, H, W
    print(video_input_ids.shape) # 1 D
    print(video_cond_mask.shape) # 1 D
    print(image[0].shape)
    print(image_input_ids.shape)
    print(image_cond_mask.shape)
    print(video_cond_mask)
    print(image_cond_mask)
