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
        self.sample_rate = args.sample_rate
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
        self.data_all = self.load_video_frames(self.data_path)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        vframes = self.data_all[index]
        total_frames = len(vframes)

        # Sampling video frames
        select_video_frames = vframes

        video_frames = []
        for path in select_video_frames:
            video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(video_frame)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)  # T C H W
        # video_clip = video_clip.transpose(0, 1)  # T C H W -> C T H W

        return video_clip, select_video_frames


    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                pass
                # print(meta[0]) # root
                # print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > 0:
                data_all.append(frames)
        self.video_num = len(data_all)
        return data_all