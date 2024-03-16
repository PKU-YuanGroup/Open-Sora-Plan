import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

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

    def __len__(self):
        # return len(self.samples)
        return 10000

    def __getitem__(self, idx):
        # video_dict = self.samples[idx]
        try:
            # video_path, text = video_dict['Filename'], video_dict['Video Description']
            # video_path = os.path.join(self.video_folder, video_path)
            #
            # video = self.tv_read(video_path)
            # video = self.transform(video)  # T C H W -> T C H W
            # video = video.transpose(0, 1)  # T C H W -> C T H W
            video = torch.rand(3, 16, 256, 256)
            text = 'ni hao'
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