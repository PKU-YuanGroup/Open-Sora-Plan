import os, yaml, pickle, shutil, tarfile, glob
import random

import cv2
import PIL
import numpy as np
import torch
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())



class WebVidTrain(Dataset):
    def __init__(self, sampling_fps, num_frames, short_size, crop_size, resampling_fps=30, **kwargs):
        self.sampling_fps = sampling_fps
        self.num_frames = num_frames
        self.sampling_duration = num_frames / sampling_fps
        self.resampling_fps = resampling_fps
        self.short_size = short_size
        self.crop_size = crop_size
        path_list = '/remote-home/yeyang/WebVid-2.5m/train/path_list.txt'
        with open(path_list, 'r') as f:
            paths = f.readlines()
        self.base = [i.strip().replace('/data/', '/data/videos/') for i in paths][:400000]

        self.transform = Compose(
        [
            # UniformTemporalSubsample(num_frames),
            Lambda(lambda x: (2 * (x / 255.0) - 1)),
            # NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ShortSideScale(size=self.short_size),
            RandomCropVideo(size=self.crop_size),
            # RandomHorizontalFlipVideo(p=0.5),
        ]
    )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        try:
            video_path = self.base[i]
            video_data = self.read_video(video_path)
            video_outputs = self.transform(video_data)
            # video_outputs = torch.rand(3, 16, 128, 128)
            return video_outputs
        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def read_video(self, video_path):
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        # fps_vid = decord_vr.get_avg_fps()
        # if fps_vid is None or fps_vid >= 40 or fps_vid <= 20:
        #     raise ValueError(f'fps is {fps_vid}')

        total_frames = len(decord_vr)
        sampling_total_frames = int(self.sampling_duration * self.resampling_fps)
        s = random.randint(0, total_frames - sampling_total_frames - 1)
        e = s + sampling_total_frames

        frame_id_list = np.linspace(s, e - 1, self.num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return video_data

class WebVidValidation(Dataset):
    def __init__(self, sampling_fps, num_frames, short_size, crop_size, resampling_fps=30, **kwargs):
        self.sampling_fps = sampling_fps
        self.num_frames = num_frames
        self.sampling_duration = num_frames / sampling_fps
        self.resampling_fps = resampling_fps
        self.short_size = short_size
        self.crop_size = crop_size
        path_list = '/remote-home/yeyang/WebVid-2.5m/train/path_list.txt'
        with open(path_list, 'r') as f:
            paths = f.readlines()
        self.base = [i.strip().replace('/data/', '/data/videos/') for i in paths][400000:]

        self.transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: (2 * (x / 255.0) - 1)),
                # NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=self.short_size),
                RandomCropVideo(size=self.crop_size),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        try:
            video_path = self.base[i]
            video_data = self.read_video(video_path)
            video_outputs = self.transform(video_data)
            # video_outputs = torch.rand(3, 16, 128, 128)
            return video_outputs
        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def read_video(self, video_path):
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        # fps_vid = decord_vr.get_avg_fps()
        # if fps_vid is None or fps_vid >= 40 or fps_vid <= 20:
        #     raise ValueError(f'fps is {fps_vid}')

        total_frames = len(decord_vr)
        sampling_total_frames = int(self.sampling_duration * self.resampling_fps)
        s = random.randint(0, total_frames - sampling_total_frames - 1)
        e = s + sampling_total_frames

        frame_id_list = np.linspace(s, e - 1, self.num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return video_data
