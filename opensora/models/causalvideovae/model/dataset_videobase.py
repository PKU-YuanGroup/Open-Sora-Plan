import os.path as osp
import random
from glob import glob

from torchvision import transforms
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Lambda

from ..dataset.transform import ToTensorVideo, CenterCropVideo
from ..utils.dataset_utils import DecordInit

def TemporalRandomCrop(total_frames, size):
    """
    Performs a random temporal crop on a video sequence.

    This function randomly selects a continuous frame sequence of length `size` from a video sequence.
    `total_frames` indicates the total number of frames in the video sequence, and `size` represents the length of the frame sequence to be cropped.

    Parameters:
    - total_frames (int): The total number of frames in the video sequence.
    - size (int): The length of the frame sequence to be cropped.

    Returns:
    - (int, int): A tuple containing two integers. The first integer is the starting frame index of the cropped sequence,
                  and the second integer is the ending frame index (inclusive) of the cropped sequence.
    """
    rand_end = max(0, total_frames - size - 1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + size, total_frames)
    return begin_index, end_index

def resize(x, resolution):
    height, width = x.shape[-2:]
    resolution = min(2 * resolution, height, width)
    aspect_ratio = width / height
    if width <= height:
        new_width = resolution
        new_height = int(resolution / aspect_ratio)
    else:
        new_height = resolution
        new_width = int(resolution * aspect_ratio)
    resized_x = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=True, antialias=True)
    return resized_x

class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    video_exts = ['avi', 'mp4', 'webm']
    def __init__(self, video_folder, sequence_length, image_folder=None, train=True, resolution=64, sample_rate=1, dynamic_sample=True):

        self.train = train
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.v_decoder = DecordInit()
        self.video_folder = video_folder
        self.dynamic_sample = dynamic_sample

        self.transform = transforms.Compose([
            ToTensorVideo(),
            # Lambda(lambda x: resize(x, self.resolution)),
            CenterCropVideo(self.resolution),
            Lambda(lambda x: 2.0 * x - 1.0)
        ])
        print('Building datasets...')
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        samples += sum([glob(osp.join(self.video_folder, '**', f'*.{ext}'), recursive=True)
                            for ext in self.video_exts], [])
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
            print(f'Error with {e}, {video_path}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

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
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sequence_length, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data