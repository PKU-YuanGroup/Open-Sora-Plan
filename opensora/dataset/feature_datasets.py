import os
import torch
import random
import torch.utils.data as data

import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


class LandscopeFeatures(data.Dataset):
    def __init__(self, args, temporal_sample):

        self.args = args
        self.data_path = args.data_path
        self.temporal_sample = temporal_sample

        data_all = list(glob(self.data_path))
        self.num_frames = self.args.num_frames
        print('Building dataset...')
        self.data_all = [i for i in tqdm(data_all) if self.num_frames < np.load(i).shape[1]]
        print(f'Total {len(self.data_all)} to train, origin dataset have {len(data_all)} videos.')



    def __getitem__(self, index):
        try:
            # import ipdb;ipdb.set_trace()
            npy_path = self.data_all[index]
            vframes = np.load(npy_path)[0]  # f, c, h, w

            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert end_frame_ind - start_frame_ind >= self.target_video_len
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
            video_clip = vframes[frame_indice]  #
            video_clip = torch.from_numpy(video_clip)
            return video_clip, 1
        except Exception as e:
            print(f'Error with {e}', npy_path)
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def __len__(self):
        return len(self.data_all)

