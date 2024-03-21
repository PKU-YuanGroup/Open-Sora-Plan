import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


#
# class T2V_dataset(Dataset):
#     def __init__(self, args, transform, temporal_sample, tokenizer):
#
#         with open(args.data_path, 'r') as csvfile:
#             self.samples = list(csv.DictReader(csvfile))
#         self.video_folder = args.video_folder
#         self.num_frames = args.num_frames
#         self.transform = transform
#         self.temporal_sample = temporal_sample
#         self.tokenizer = tokenizer
#         self.model_max_length = args.model_max_length
#         self.v_decoder = DecordInit()
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         video_dict = self.samples[idx]
#         try:
#             video_path, text = video_dict['Filename'], video_dict['Video Description']
#             video_path = os.path.join(self.video_folder, video_path)
#
#             video = self.tv_read(video_path)
#             video = self.transform(video)  # T C H W -> T C H W
#             video = video.transpose(0, 1)  # T C H W -> C T H
#
#
#             text = text_preprocessing(text)
#             text_tokens_and_mask = self.tokenizer(
#                 text,
#                 max_length=self.model_max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_attention_mask=True,
#                 add_special_tokens=True,
#                 return_tensors='pt'
#             )
#
#             input_ids = text_tokens_and_mask['input_ids']
#             cond_mask = text_tokens_and_mask['attention_mask']
#
#             return video, input_ids, cond_mask
#         except Exception as e:
#             print(f'Error with {e}, {video_path}')
#             return self.__getitem__(random.randint(0, self.__len__()-1))
#
#     def tv_read(self, path):
#         vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
#         total_frames = len(vframes)
#
#         # Sampling video frames
#         start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
#         # assert end_frame_ind - start_frame_ind >= self.num_frames
#         frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
#         video = vframes[frame_indice]  # (T, C, H, W)
#
#         return video
#
#     def decord_read(self, path):
#         decord_vr = self.v_decoder(path)
#         total_frames = len(decord_vr)
#         # Sampling video frames
#         start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
#         # assert end_frame_ind - start_frame_ind >= self.num_frames
#         frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
#
#         video_data = decord_vr.get_batch(frame_indice).asnumpy()
#         video_data = torch.from_numpy(video_data)
#         video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
#         return video_data


class T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):

        # with open(args.data_path, 'r') as csvfile:
        #     self.samples = list(csv.DictReader(csvfile))
        self.video_folder = args.video_folder
        self.num_frames = args.num_frames
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.v_decoder = DecordInit()
        self.video_path = []
        for x in tqdm(os.listdir('/remote-home/yeyang/panda2m')):
            if x.isdigit():
                x = os.path.join('/remote-home/yeyang/panda2m', x)
                for y in os.listdir(x):
                    if y[-3] == 'm':
                        self.video_path.append(os.path.join(x, y))
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        if self.use_image_num != 0 and not self.use_img_from_vid:
            self.img_cap_list = self.get_img_cap_list()

    def __len__(self):
        # return len(self.samples)
        return len(self.video_path)

    def __getitem__(self, idx):
        try:
            video = self.tv_read(self.video_path[idx])
            video = self.transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            with open(self.video_path[idx].replace('mp4', 'txt'), 'r', encoding='utf-8') as file:
                text = file.read()

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
            input_ids = text_tokens_and_mask['input_ids'].squeeze(0)
            cond_mask = text_tokens_and_mask['attention_mask'].squeeze(0)

            if self.use_image_num != 0 and self.use_img_from_vid:
                select_image_idx = np.linspace(0, self.num_frames-1, self.use_image_num, dtype=int)
                assert self.num_frames >= self.use_image_num
                images = video[:, select_image_idx]  # c, num_img, h, w
                video = torch.cat([video, images], dim=1)  # c, num_frame+num_img, h, w
                input_ids = torch.stack([input_ids] * (1+self.use_image_num))  # 1+self.use_image_num, l
                cond_mask = torch.stack([cond_mask] * (1+self.use_image_num))  # 1+self.use_image_num, l
            elif self.use_image_num != 0 and not self.use_img_from_vid:
                images, captions = self.img_cap_list[idx]
                raise NotImplementedError
            else:
                pass

            return video, input_ids, cond_mask
        except Exception as e:
            # print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def tv_read(self, path):
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

        video = vframes[frame_indice]  # (T, C, H, W)

        return video

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def get_img_cap_list(self):
        raise NotImplementedError