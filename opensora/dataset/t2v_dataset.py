import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from opensora.utils.utils import text_preprocessing

class T2V_dataset(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=512, sample_stride=4, sample_n_frames=16,
            is_image=False,
            is_uniform=False,
        ):
        
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.is_uniform      = is_uniform
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.cached_indices = {}

    def _generate_frame_indices(self, video_length, n_frames):
        if self.is_uniform:
            if video_length <= n_frames:
                indices = list(range(video_length))
                additional_frames_needed = n_frames - video_length
                processed_extra_frames = 0
                current_length = len(indices)
                while processed_extra_frames < additional_frames_needed:
                    repeat_round = min(additional_frames_needed - processed_extra_frames, current_length)
                    for i in range(repeat_round):
                        index_to_repeat = (i * video_length) // repeat_round
                        indices.append(indices[index_to_repeat])
                        processed_extra_frames += 1
                    current_length = len(indices)
                return sorted(indices[:n_frames])
            else:
                interval = (video_length - 1) / (n_frames - 1)
                indices = [int(round(i * interval)) for i in range(n_frames)]
                indices[-1] = video_length - 1
                return indices
        else:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            return np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int).tolist()
        
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_path, text = video_dict['Filename'], video_dict['Video Description']
        video_dir    = os.path.join(self.video_folder, video_path)
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        #make full use of the video thus not cache
        """
        if videoid not in self.cached_indices:
            self.cached_indices[videoid] = self._generate_frame_indices(video_length, self.sample_n_frames) if not self.is_image else [random.randint(0, video_length - 1)]
            # print(len(self.cached_indices))

        batch_index = self.cached_indices[videoid]
        """
        batch_index = self._generate_frame_indices(video_length, self.sample_n_frames) if not self.is_image else [random.randint(0, video_length - 1)]
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2) / 255.
        del video_reader  # Release resources
        
        if self.is_image:
            pixel_values = pixel_values[0]
        text = text_preprocessing(text)
        return pixel_values, text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, text = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(videos=pixel_values, text=text)
        return sample