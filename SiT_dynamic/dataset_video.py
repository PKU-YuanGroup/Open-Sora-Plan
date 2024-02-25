import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
import random

class NormalizeVideo(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeVideo, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, video):
        c, t, h, w = video.shape
        # 创建一个新的张量来存储标准化后的帧
        normalized_video = torch.zeros_like(video)
        # 在时间维度上迭代每一帧
        for i in range(t):
            frame = video[:, i, :, :]
            # 应用标准化到当前帧
            normalized_frame = normalize(frame, self.mean, self.std)
            normalized_video[:, i, :, :] = normalized_frame
        return normalized_video

class UCF101ClassConditionedDataset(Dataset):
    def __init__(self, root_dir, sample_size=256, frames_per_clip=16, step_between_clips=4, frame_rate=None):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.frame_rate = frame_rate
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        dataset = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.endswith('.avi'):
                    item = (os.path.join(class_path, fname), self.class_to_idx[class_name])
                    dataset.append(item)
        return dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _, info = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
        total_frames = video.size(0)
        
        start_frames = range(0, total_frames - self.frames_per_clip * self.step_between_clips, self.step_between_clips)
        
        if len(start_frames) > 0:
            start_frame = random.choice(start_frames)
        else:
            start_frame = 0
        
        frames = []
        for i in range(self.frames_per_clip):
            frame_idx = start_frame + i * self.step_between_clips
            if frame_idx < total_frames:
                frames.append(video[frame_idx])
            else:
                # Handle the case where there aren't enough frames; for example, repeat the last frame
                frames.append(video[-1])  # This line is a simple strategy; adapt as needed

        # Ensure the clip is created outside the loop and only once, after collecting all frames
        clip = torch.stack(frames).float() / 255
        clip = clip.permute(3, 0, 1, 2)  # Change from T H W C to C T H W

        if self.transform is not None:
            clip = self.transform(clip)

        return clip, label
