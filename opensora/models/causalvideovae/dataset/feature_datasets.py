import json
import os
import torch
import random
import torch.utils.data as data

import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .transform import center_crop, RandomCropVideo
from ..utils.dataset_utils import DecordInit


class T2V_Feature_dataset(Dataset):
    def __init__(self, args, temporal_sample):

        self.video_folder = args.video_folder
        self.num_frames = args.video_length
        self.temporal_sample = temporal_sample

        print('Building dataset...')
        if os.path.exists('samples_430k.json'):
            with open('samples_430k.json', 'r') as f:
                self.samples = json.load(f)
        else:
            self.samples = self._make_dataset()
            with open('samples_430k.json', 'w') as f:
                json.dump(self.samples, f, indent=2)

        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        if self.use_image_num != 0 and not self.use_img_from_vid:
            self.img_cap_list = self.get_img_cap_list()

    def _make_dataset(self):
        all_mp4 = list(glob(os.path.join(self.video_folder, '**', '*.mp4'), recursive=True))
        # all_mp4 = all_mp4[:1000]
        samples = []
        for i in tqdm(all_mp4):
            video_id = os.path.basename(i).split('.')[0]
            ae = os.path.split(i)[0].replace('data_split_tt', 'lb_causalvideovae444_feature')
            ae = os.path.join(ae, f'{video_id}_causalvideovae444.npy')
            if not os.path.exists(ae):
                continue

            t5 = os.path.split(i)[0].replace('data_split_tt', 'lb_t5_feature')
            cond_list = []
            cond_llava = os.path.join(t5, f'{video_id}_t5_llava_fea.npy')
            mask_llava = os.path.join(t5, f'{video_id}_t5_llava_mask.npy')
            if os.path.exists(cond_llava) and os.path.exists(mask_llava):
                llava = dict(cond=cond_llava, mask=mask_llava)
                cond_list.append(llava)
            cond_sharegpt4v = os.path.join(t5, f'{video_id}_t5_sharegpt4v_fea.npy')
            mask_sharegpt4v = os.path.join(t5, f'{video_id}_t5_sharegpt4v_mask.npy')
            if os.path.exists(cond_sharegpt4v) and os.path.exists(mask_sharegpt4v):
                sharegpt4v = dict(cond=cond_sharegpt4v, mask=mask_sharegpt4v)
                cond_list.append(sharegpt4v)
            if len(cond_list) > 0:
                sample = dict(ae=ae, t5=cond_list)
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # try:
        sample = self.samples[idx]
        ae, t5 = sample['ae'], sample['t5']
        t5 = random.choice(t5)
        video_origin = np.load(ae)[0]  # C T H W
        _, total_frames, _, _ = video_origin.shape
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.num_frames
        select_video_idx = np.linspace(start_frame_ind, end_frame_ind - 1, num=self.num_frames, dtype=int)  # start, stop, num=50
        # print('select_video_idx', total_frames, select_video_idx)
        video = video_origin[:, select_video_idx]  # C num_frames H W
        video = torch.from_numpy(video)

        cond = torch.from_numpy(np.load(t5['cond']))[0]  # L
        cond_mask = torch.from_numpy(np.load(t5['mask']))[0]  # L D

        if self.use_image_num != 0 and self.use_img_from_vid:
            select_image_idx = np.random.randint(0, total_frames, self.use_image_num)
            # print('select_image_idx', total_frames, self.use_image_num, select_image_idx)
            images = video_origin[:, select_image_idx]  # c, num_img, h, w
            images = torch.from_numpy(images)
            video = torch.cat([video, images], dim=1)  # c, num_frame+num_img, h, w
            cond = torch.stack([cond] * (1+self.use_image_num))  # 1+self.use_image_num, l
            cond_mask = torch.stack([cond_mask] * (1+self.use_image_num))  # 1+self.use_image_num, l
        elif self.use_image_num != 0 and not self.use_img_from_vid:
            images, captions = self.img_cap_list[idx]
            raise NotImplementedError
        else:
            pass

        return video, cond, cond_mask
        # except Exception as e:
        #     print(f'Error with {e}, {sample}')
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_img_cap_list(self):
        raise NotImplementedError




class T2V_T5_Feature_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample):

        self.video_folder = args.video_folder
        self.num_frames = args.num_frames
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.v_decoder = DecordInit()

        print('Building dataset...')
        if os.path.exists('samples_430k.json'):
            with open('samples_430k.json', 'r') as f:
                self.samples = json.load(f)
                self.samples = [dict(ae=i['ae'].replace('lb_causalvideovae444_feature', 'data_split_1024').replace('_causalvideovae444.npy', '.mp4'), t5=i['t5']) for i in self.samples]
        else:
            self.samples = self._make_dataset()
            with open('samples_430k.json', 'w') as f:
                json.dump(self.samples, f, indent=2)

        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        if self.use_image_num != 0 and not self.use_img_from_vid:
            self.img_cap_list = self.get_img_cap_list()

    def _make_dataset(self):
        all_mp4 = list(glob(os.path.join(self.video_folder, '**', '*.mp4'), recursive=True))
        # all_mp4 = all_mp4[:1000]
        samples = []
        for i in tqdm(all_mp4):
            video_id = os.path.basename(i).split('.')[0]
            # ae = os.path.split(i)[0].replace('data_split', 'lb_causalvideovae444_feature')
            # ae = os.path.join(ae, f'{video_id}_causalvideovae444.npy')
            ae = i
            if not os.path.exists(ae):
                continue

            t5 = os.path.split(i)[0].replace('data_split_1024', 'lb_t5_feature')
            cond_list = []
            cond_llava = os.path.join(t5, f'{video_id}_t5_llava_fea.npy')
            mask_llava = os.path.join(t5, f'{video_id}_t5_llava_mask.npy')
            if os.path.exists(cond_llava) and os.path.exists(mask_llava):
                llava = dict(cond=cond_llava, mask=mask_llava)
                cond_list.append(llava)
            cond_sharegpt4v = os.path.join(t5, f'{video_id}_t5_sharegpt4v_fea.npy')
            mask_sharegpt4v = os.path.join(t5, f'{video_id}_t5_sharegpt4v_mask.npy')
            if os.path.exists(cond_sharegpt4v) and os.path.exists(mask_sharegpt4v):
                sharegpt4v = dict(cond=cond_sharegpt4v, mask=mask_sharegpt4v)
                cond_list.append(sharegpt4v)
            if len(cond_list) > 0:
                sample = dict(ae=ae, t5=cond_list)
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            ae, t5 = sample['ae'], sample['t5']
            t5 = random.choice(t5)

            video = self.decord_read(ae)
            video = self.transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            total_frames = video.shape[1]
            cond = torch.from_numpy(np.load(t5['cond']))[0]  # L
            cond_mask = torch.from_numpy(np.load(t5['mask']))[0]  # L D

            if self.use_image_num != 0 and self.use_img_from_vid:
                select_image_idx = np.random.randint(0, total_frames, self.use_image_num)
                # print('select_image_idx', total_frames, self.use_image_num, select_image_idx)
                images = video.numpy()[:, select_image_idx]  # c, num_img, h, w
                images = torch.from_numpy(images)
                video = torch.cat([video, images], dim=1)  # c, num_frame+num_img, h, w
                cond = torch.stack([cond] * (1+self.use_image_num))  # 1+self.use_image_num, l
                cond_mask = torch.stack([cond_mask] * (1+self.use_image_num))  # 1+self.use_image_num, l
            elif self.use_image_num != 0 and not self.use_img_from_vid:
                images, captions = self.img_cap_list[idx]
                raise NotImplementedError
            else:
                pass

            return video, cond, cond_mask
        except Exception as e:
            print(f'Error with {e}, {sample}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def get_img_cap_list(self):
        raise NotImplementedError