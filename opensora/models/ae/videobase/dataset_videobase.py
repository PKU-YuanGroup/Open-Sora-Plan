import os
import os.path as osp
import math
import glob
import pickle
import random
import warnings

import torch
import decord
import torchvision
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from decord import VideoReader, cpu
from torchvision.datasets.video_utils import VideoClips

from .vqvae.dataset_vqvae import VQVAEDataset
from .causal_vqvae.dataset_causalvqvae import CausalVQVAEDataset

decord.bridge.set_bridge('torch')


def build_videoae_dataset(video_folder, image_folder, sequence_length, resolution, train=True):
    if image_folder is not None:
        return CausalVQVAEDataset(video_folder, sequence_length, image_folder=image_folder, resolution=resolution, train=train)
    elif 'kinetics' in video_folder:
        return Kinetics400Dataset(video_folder, sequence_length, resolution=resolution, train=train)
    else:
        return VQVAEDataset(video_folder, sequence_length, resolution=resolution, train=train)


class Kinetics400Dataset(data.Dataset):
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, resolution=64, train=True):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        if train:
            folder = osp.join(data_folder, 'videos_train')
            file_list = osp.join(data_folder, 'kinetics400_train_list_videos.txt' if train else 'kinetics400_val_list_videos.txt')
        else:
            folder = osp.join(data_folder, 'videos_val')
            file_list = osp.join(data_folder, 'kinetics400_val_list_videos.txt')

        self.files = [os.path.join(folder, x.split(' ')[0]) for x in open(file_list, 'r').readlines()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        resolution = self.resolution
        decord_vr = VideoReader(self.files[idx], ctx=cpu(0))

        if len(decord_vr) < self.sequence_length:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        start_idx = random.randint(0, len(decord_vr) - self.sequence_length)
        video_data = decord_vr.get_batch(np.arange(start_idx, start_idx + self.sequence_length, 1))

        return dict(video=preprocess(video_data, resolution))


# Copied from https://github.com/wilson1yan/VideoGPT
class VideoAEDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    video_exts = ['avi', 'mp4', 'webm']
    image_exts = ['png', 'jpg', 'jpeg']
    def __init__(self, video_folder, sequence_length, image_folder=None, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        if image_folder is not None:
            raise NotImplementedError("Image training is not supported now.")
        
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        files = []
        video_files = []
        image_files = []
        for data_folder in [video_folder, image_folder]:
            if data_folder is None:
                continue
            folder = data_folder
            video_files += sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                         for ext in self.video_exts], [])
            image_files += sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                         for ext in self.image_exts], [])
        files = video_files + image_files
        # hacky way to compute # of classes (count # of unique parent directories)
        # self.classes = list(set([get_parent_dir(f) for f in files]))
        # self.classes.sort()
        # self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        if len(video_files) != 0:
            cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
            if not osp.exists(cache_file):
                clips = VideoClips(video_files, sequence_length, num_workers=32)
                if dist.is_initialized() and dist.get_rank() == 0:
                    pickle.dump(clips.metadata, open(cache_file, 'wb'))
            else:
                metadata = pickle.load(open(cache_file, 'rb'))
                clips = VideoClips(video_files, sequence_length,
                                   _precomputed_metadata=metadata)

            self._clips = clips
            self._clips_num = self._clips.num_clips()
        else:
            self._clips = None
            self._clips_num = 0
        self.image_files = image_files

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips_num + len(self.image_files)

    def __getitem__(self, idx):
        resolution = self.resolution
        if idx < self._clips_num:
            video, _, _, idx = self._clips.get_clip(idx)
            video = preprocess(video, resolution)
            class_name = get_parent_dir(self._clips.video_paths[idx])
        else:
            idx -= self._clips_num
            image = Image.open(self.image_files[idx])
            video = preprocess_image(image, resolution, self.sequence_length)
        # label = self.class_to_label[class_name]
        return dict(video=video, label="")


# Copied from https://github.com/wilson1yan/VideoGPT
def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


# Copied from https://github.com/wilson1yan/VideoGPT
def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video


def preprocess_image(image, resolution, sequence_length=1):
    # image: HWC, {0, ..., 255}
    image = image.convert("RGB")
    w,h = image.size
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    image = image.resize(target_size)
    
    image = transforms.ToTensor()(image)
    image = image.float()
    c, h, w = image.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    image = image[:, h_start:h_start + resolution, w_start:w_start + resolution]
    image -= 0.5
    c, h, w = image.shape
    new_image = torch.zeros((c, sequence_length, h, w))
    new_image = new_image.to(image.device)
    new_image[:, :1, :, :] = image.unsqueeze(1)
    new_image = new_image.contiguous()
    
    return new_image
