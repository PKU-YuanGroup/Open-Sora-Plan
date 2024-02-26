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

class PadTemporal(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int
    ):
        super().__init__()
        self._size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert len(x.shape) == 4
        assert x.dtype == torch.float32
        c, t, h, w = x.shape
        if t < self._size:
            pad = torch.zeros(c, self._size - t, h, w)
            x = torch.cat([x, pad], dim=1)
        return x


class UCF101(Dataset):
    def __init__(self, root_dir, sample_rate, num_frames, short_size, crop_size, **kwargs):
        self.root_dir = root_dir

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.video_paths = self._make_dataset()

        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.sample_frames_len = self.sample_rate * self.num_frames
        self.short_size = short_size
        self.crop_size = crop_size

        self.transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: (2 * (x / 255.0) - 1)),
                # NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=self.short_size),
                RandomCropVideo(size=self.crop_size),
                PadTemporal(size=self.num_frames),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, i):
        try:
            video_path = self.video_paths[i]
            video_data = self.read_video(video_path)
            video_outputs = self.transform(video_data)
            # video_outputs = torch.rand(3, 16, 128, 128)
            return video_outputs
        except Exception as e:
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def read_video(self, video_path):
        decord_vr = VideoReader(video_path, ctx=cpu(0))


        total_frames = len(decord_vr)
        if total_frames > self.sample_frames_len:
            s = random.randint(0, total_frames - self.sample_frames_len - 1)
            e = s + self.sample_frames_len
            num_frames = self.num_frames
        else:
            s = 0
            e = total_frames
            num_frames = int(total_frames / self.sample_frames_len * self.num_frames)
            # print(f'sample_frames_len {self.sample_frames_len}, only can {num_frames*self.sample_rate}', video_path, total_frames)


        # random drop to dynamic input frames
        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        return video_data


    def _make_dataset(self):
        dataset = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith('.avi'):
                    item = os.path.join(class_path, fname)
                    dataset.append(item)
        return dataset

