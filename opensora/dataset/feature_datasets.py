import os
import torch
import random
import torch.utils.data as data

import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

from opensora.dataset.transform import center_crop, RandomCropVideo


class LandscopeVideoFeature(data.Dataset):
    def __init__(self, args):

        self.args = args
        self.data_path = args.data_path
        self.max_image_size = args.latent_size

        data_all = list(glob(self.data_path))
        self.num_frames = self.args.num_frames
        self.sample_rate = self.args.sample_rate
        print('Building dataset...')
        self.data_all = [i for i in tqdm(data_all) if self.num_frames // args.ae_stride_t == np.load(i).shape[0]]
        print(f'Total {len(self.data_all)} to train, origin dataset have {len(data_all)} videos.')

    def __getitem__(self, index):
        # try:
        npy_path = self.data_all[index]
        vframes = np.load(npy_path)  # t, c, h, w
        video_clip = torch.from_numpy(vframes)  # T C H W
        return video_clip, 1
        # except Exception as e:
        #     print(f'Error with {e}', npy_path)
        #     return self.__getitem__(random.randint(0, self.__len__()-1))

    def __len__(self):
        return len(self.data_all)


class LandscopeFeature(data.Dataset):
    def __init__(self, args, temporal_sample):

        self.args = args
        self.data_path = args.data_path
        self.max_image_size = args.latent_size
        self.temporal_sample = temporal_sample

        data_all = list(glob(self.data_path))
        self.num_frames = self.args.num_frames
        self.sample_rate = self.args.sample_rate
        print('Building dataset...')
        self.data_all = [i for i in tqdm(data_all) if self.num_frames * self.sample_rate < np.load(i).shape[0]]
        print(f'Total {len(self.data_all)} to train, origin dataset have {len(data_all)} videos.')

    def __getitem__(self, index):
        try:
            npy_path = self.data_all[index]
            vframes = np.load(npy_path)  # t, c, h, w

            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert end_frame_ind - start_frame_ind >= self.num_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.num_frames, dtype=int) # start, stop, num=50
            video_clip = vframes[frame_indice[0]: frame_indice[-1]+1: self.sample_rate]   #
            video_clip = torch.from_numpy(video_clip)  # T C H W
            return video_clip, 1
        except Exception as e:
            print(f'Error with {e}', npy_path)
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def __len__(self):
        return len(self.data_all)


class SkyFeature(data.Dataset):
    def __init__(self, args, temporal_sample=None):

        self.args = args
        self.data_path = args.data_path
        self.temporal_sample = temporal_sample
        self.num_frames = self.args.num_frames
        self.sample_rate = self.args.sample_rate
        self.data_all = self.load_video_frames(self.data_path)

    def __getitem__(self, index):

        try:
            vframes = self.data_all[index]
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert end_frame_ind - start_frame_ind >= self.num_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num=self.num_frames, dtype=int)  # start, stop, num=50

            select_video_frames = vframes[frame_indice[0]: frame_indice[-1] + 1: self.sample_rate]

            video_frames = []
            for path in select_video_frames:
                video_frame = torch.from_numpy(np.load(path)).unsqueeze(0)  # 1 c h w
                video_frames.append(video_frame)
            video_clip = torch.cat(video_frames, dim=0)  # T C H W
            # video_clip = video_clip.transpose(0, 1)  # T C H W -> C T H W

            return video_clip, 1
        except Exception as e:
            print(f'Error with {e}', vframes)
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def __len__(self):
        return self.video_num

    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[2]))
            except:
                pass
                # print(meta[0]) # root
                # print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if item.endswith('.npy')]
            if len(frames) > max(0, self.num_frames * self.sample_rate):  # need all > (16 * frame-interval) videos
                # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)
        self.video_num = len(data_all)
        return data_all