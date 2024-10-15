import os.path as osp
import random
from glob import glob
from torchvision import transforms
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pickle
import decord
from torch.nn import functional as F
from .transform import ToTensorVideo, CenterCropVideo
from torchvision.transforms._transforms_video import CenterCropVideo as TVCenterCropVideo
from torchvision.transforms import Lambda, Compose, Resize
import torch
import os


class DecordInit(object):
    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"
            f"num_threads={self.num_threads})"
        )
        return repr_str

def TemporalRandomCrop(total_frames, size):
    rand_end = max(0, total_frames - size - 1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + size, total_frames)
    return begin_index, end_index

def _format_video_shape(video, time_compress=4, spatial_compress=8):
    """Prepare video for VAE"""
    time = video.shape[1]
    height = video.shape[2]
    width = video.shape[3]
    new_time = (
        (time - (time - 1) % time_compress) if (time - 1) % time_compress != 0 else time
    )
    new_height = (
        (height - (height) % spatial_compress)
        if height % spatial_compress != 0
        else height
    )
    new_width = (
        (width - (width) % spatial_compress) if width % spatial_compress != 0 else width
    )
    return video[:, :new_time, :new_height, :new_width]


class TrainVideoDataset(data.Dataset):
    video_exts = ["avi", "mp4", "webm"]

    def __init__(
        self,
        video_folder,
        sequence_length,
        train=True,
        resolution=64,
        sample_rate=1,
        dynamic_sample=True,
        cache_file=None,
        is_main_process=False,
    ):

        self.train = train
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.v_decoder = DecordInit()
        self.video_folder = video_folder
        self.dynamic_sample = dynamic_sample
        self.cache_file = cache_file
        self.transform = transforms.Compose(
            [
                ToTensorVideo(),
                Resize(self.resolution),
                CenterCropVideo(self.resolution),
                Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        print("Building datasets...")
        self.is_main_process = is_main_process
        self.samples = self._make_dataset()

    def _make_dataset(self):
        cache_file = osp.join(self.video_folder, self.cache_file)

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            samples += sum(
                [
                    glob(osp.join(self.video_folder, "**", f"*.{ext}"), recursive=True)
                    for ext in self.video_exts
                ],
                [],
            )
            if self.is_main_process:
                with open(cache_file, "wb") as f:
                    pickle.dump(samples, f)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        try:
            video = self.decord_read(video_path)
            video = self.transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            return dict(video=video, label="")
        except Exception as e:
            print(f"Error with {e}, {video_path}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if self.dynamic_sample:
            sample_rate = random.randint(1, self.sample_rate)
        else:
            sample_rate = self.sample_rate
        size = self.sequence_length * sample_rate
        start_frame_ind, end_frame_ind = TemporalRandomCrop(total_frames, size)
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sequence_length, dtype=int
        )

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)
        return video_data

def resize(x, resolution):
    height, width = x.shape[-2:]
    aspect_ratio = width / height
    if width <= height:
        new_width = resolution
        new_height = int(resolution / aspect_ratio)
    else:
        new_height = resolution
        new_width = int(resolution * aspect_ratio)
    resized_x = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True, antialias=True)
    return resized_x

class ValidVideoDataset(data.Dataset):
    video_exts = ["avi", "mp4", "webm"]
    
    def __init__(
        self,
        real_video_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
        is_main_process=False
    ) -> None:
        super().__init__()
        self.is_main_process = is_main_process
        self.real_video_files = self._make_dataset(real_video_dir)
        
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution
        self.v_decoder = DecordInit()
        self.transform = Compose(
            [
                ToTensorVideo(),
                Resize(resolution),
                CenterCropVideo(resolution) if crop_size is not None else Lambda(lambda x: x),
            ]
        )
        
    def _make_dataset(self, real_video_dir):
        cache_file = osp.join(real_video_dir, "idx.pkl")

        if osp.exists(cache_file):
            with open(cache_file, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = []
            samples += sum(
                [
                    glob(osp.join(real_video_dir, "**", f"*.{ext}"), recursive=True)
                    for ext in self.video_exts
                ],
                [],
            )
            if self.is_main_process:
                with open(cache_file, "wb") as f:
                    pickle.dump(samples, f)
        return samples
    
    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        try:
            if index >= len(self):
                raise IndexError
            real_video_file = self.real_video_files[index]
            real_video_tensor = self._load_video(real_video_file)
            real_video_tensor = self.transform(real_video_tensor)
            video_name = os.path.basename(real_video_file)
            return {'video': real_video_tensor, 'file_name': video_name }
        except:
            print(f"Video error: {self.real_video_files[index]}")
            return self.__getitem__(0)

    def _load_video(self, video_path, sample_rate=None):
        num_frames = self.num_frames
        if not sample_rate:
            sample_rate = self.sample_rate
        try:
            decord_vr = self.v_decoder(video_path)
        except:
            raise Exception(f"fail to load {video_path}.")
        total_frames = len(decord_vr)
        sample_frames_len = sample_rate * num_frames

        if total_frames >= sample_frames_len:
            s = 0
            e = s + sample_frames_len
            num_frames = num_frames
        else:
            raise Exception("video too short!")
            
        frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data
