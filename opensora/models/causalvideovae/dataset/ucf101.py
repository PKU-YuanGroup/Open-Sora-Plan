import math
import os

import decord
import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torch.nn import functional as F
import random

from ..utils.dataset_utils import DecordInit


class UCF101(Dataset):
    def __init__(self, args, transform, temporal_sample):
        self.data_path = args.data_path
        self.num_frames = args.num_frames
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.v_decoder = DecordInit()

        self.classes = sorted(os.listdir(self.data_path))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()


    def _make_dataset(self):
        dataset = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith('.avi'):
                    item = (os.path.join(class_path, fname), self.class_to_idx[class_name])
                    dataset.append(item)
        return dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            video = self.tv_read(video_path)
            video = self.transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            return video, label
        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

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

