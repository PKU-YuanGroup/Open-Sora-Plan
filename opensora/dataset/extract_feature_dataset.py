import os
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from opensora.utils.dataset_utils import DecordInit, is_image_file


class ExtractVideo2Feature(Dataset):
    def __init__(self, args, transform):
        self.data_path = args.data_path
        self.transform = transform
        self.v_decoder = DecordInit()
        self.samples = list(glob(f'{self.data_path}'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        video = self.decord_read(video_path)
        video = self.transform(video)  # T C H W -> T C H W
        return video, video_path

    def tv_read(self, path):
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        frame_indice = list(range(total_frames))
        video = vframes[frame_indice]
        return video

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        frame_indice = list(range(total_frames))
        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data



class ExtractImage2Feature(Dataset):
    def __init__(self, args, transform):
        self.data_path = args.data_path
        self.transform = transform
        self.data_all = list(glob(f'{self.data_path}'))

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        path = self.data_all[index]
        video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
        video_frame = video_frame.permute(0, 3, 1, 2)
        video_frame = self.transform(video_frame)  # T C H W
        # video_frame = video_frame.transpose(0, 1)  # T C H W -> C T H W

        return video_frame, path

