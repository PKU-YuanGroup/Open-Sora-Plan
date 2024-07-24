import time
import traceback

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import glob
import json
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from decord import VideoReader
from os.path import join as opj
from collections import Counter

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
logger = get_logger(__name__)

def filter_json_by_existed_files(directory, data, postfixes=[".mp4", ".jpg"]):
    # 构建搜索模式，以匹配指定后缀的文件
    matching_files = []
    for postfix in postfixes:
        pattern = os.path.join(directory, '**', f'*{postfix}')
        matching_files.extend(glob.glob(pattern, recursive=True))

    # 使用文件的绝对路径构建集合
    mp4_files_set = set(os.path.abspath(path) for path in matching_files)

    # 过滤数据条目，只保留路径在mp4文件集合中的条目
    filtered_items = [item for item in data if item['path'] in mp4_files_set]

    return filtered_items


def random_video_noise(t, c, h, w):
    vid = torch.rand(t, c, h, w) * 255.0
    vid = vid.to(torch.uint8)
    return vid


class SingletonMeta(type):
    """
    这是一个元类，用于创建单例类。
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):
    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        random.shuffle(self.elements)
        print(f"n_elements: {len(self.elements)}", flush=True)

        for i in range(self.num_workers):
            self.n_used_elements[i] = 0
            per_worker = int(math.ceil(len(self.elements) / float(self.num_workers)))
            start = i * per_worker
            end = min(start + per_worker, len(self.elements))
            self.worker_elements[i] = self.elements[start: end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][self.n_used_elements[worker_id] % len(self.worker_elements[worker_id])]
        self.n_used_elements[worker_id] += 1
        return idx


dataset_prog = DataSetProg()

def find_closest_y(x, vae_stride_t=4, model_ds_t=4):
    if x < 29:
        return -1  
    for y in range(x, 12, -1):
        if (y - 1) % vae_stride_t == 0 and ((y - 1) // vae_stride_t + 1) % model_ds_t == 0:
            # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
            # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
            return y
    return -1 

def filter_resolution(h, w, max_h_div_w_ratio=17/16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False
        


class T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        self.data = args.data
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.drop_short_ratio = args.drop_short_ratio
        assert self.speed_factor >= 1
        self.v_decoder = DecordInit()

        self.support_Chinese = True
        if not ('mt5' in args.text_encoder_name):
            self.support_Chinese = False

        cap_list = self.get_cap_list()
        
        assert len(cap_list) > 0
        cap_list, self.sample_num_frames = self.define_frame_index(cap_list)
        self.lengths = self.sample_num_frames

        n_elements = len(cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list, n_elements)

        print(f"video length: {len(dataset_prog.cap_list)}", flush=True)

    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return dataset_prog.n_elements

    def __getitem__(self, idx):
        try:
            data = self.get_data(idx)
            return data
        except Exception as e:
            logger.info(f'Error with {e}')
            # 打印异常堆栈
            if idx in dataset_prog.cap_list:
                logger.info(f"Caught an exception! {dataset_prog.cap_list[idx]}")
            # traceback.print_exc()
            # traceback.print_stack()
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_data(self, idx):
        path = dataset_prog.cap_list[idx]['path']
        if path.endswith('.mp4'):
            return self.get_video(idx)
        else:
            return self.get_image(idx)
    
    def get_video(self, idx):
        # npu_config.print_msg(f"current idx is {idx}")
        # video = random.choice([random_video_noise(65, 3, 336, 448), random_video_noise(65, 3, 1024, 1024), random_video_noise(65, 3, 360, 480)])
        # # print('random shape', video.shape)
        # input_ids = torch.ones(1, 120).to(torch.long).squeeze(0)
        # cond_mask = torch.cat([torch.ones(1, 60).to(torch.long), torch.ones(1, 60).to(torch.long)], dim=1).squeeze(0)

        video_path = dataset_prog.cap_list[idx]['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        video = self.decord_read(video_path, predefine_num_frames=len(frame_indice))

        h, w = video.shape[-2:]
        assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]
        video = self.transform(video)  # T C H W -> T C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.cap_list[idx]['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text_preprocessing(text, support_Chinese=self.support_Chinese) if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        # import ipdb;ipdb.set_trace()
        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]
        # for i in image:
        #     h, w = i.shape[-2:]
        #     assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only image with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But found ratio is {round(h / w, 2)} with the shape of {i.shape}'
        
        image = self.transform_topcrop(image) if 'human_images' in image_data['path'] else self.transform(image) #  [1 C H W] -> num_img [1 C H W]

        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = image_data['cap'] if isinstance(image_data['cap'], list) else [image_data['cap']]
        caps = [random.choice(caps)]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        input_ids, cond_mask = [], []
        text = text if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']  # 1, l
        cond_mask = text_tokens_and_mask['attention_mask']  # 1, l
        return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask)

    def define_frame_index(self, cap_list):
        
        new_cap_list = []
        sample_num_frames = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_resolution_mismatch = 0
        cnt_movie = 0
        cnt_img = 0
        for i in cap_list:
            path = i['path']
            cap = i.get('cap', None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue
            if path.endswith('.mp4'):
                # ======no fps and duration=====
                duration = i.get('duration', None)
                fps = i.get('fps', None)
                if fps is None or duration is None:
                    continue

                # ======resolution mismatch=====
                resolution = i.get('resolution', None)
                if resolution is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    if resolution.get('height', None) is None or resolution.get('width', None) is None:
                        cnt_no_resolution += 1
                        continue
                    height, width = i['resolution']['height'], i['resolution']['width']
                    aspect = self.max_height / self.max_width
                    hw_aspect_thr = 1.5
                    is_pick = filter_resolution(height, width, max_h_div_w_ratio=hw_aspect_thr*aspect, 
                                                min_h_div_w_ratio=1/hw_aspect_thr*aspect)
                    if not is_pick:
                        cnt_resolution_mismatch += 1
                        continue

                    # # ignore image resolution mismatch
                    # if self.max_height > resolution['height'] or self.max_width > resolution['width']:
                    #     cnt_resolution_mismatch += 1
                    #     continue

                # import ipdb;ipdb.set_trace()
                i['num_frames'] = int(fps * duration)
                # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration. 
                if i['num_frames'] > 2.0 * (self.num_frames * fps / self.train_fps * self.speed_factor):  # too long video is not suitable for this training stage (self.num_frames)
                    cnt_too_long += 1
                    continue
                # if i['num_frames'] < 1.0/1 * (self.num_frames * fps / self.train_fps * self.speed_factor):  # too short video is not suitable for this training stage
                #     cnt_too_short += 1
                #     continue 

                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = fps / self.train_fps
                start_frame_idx = 0 
                frame_indices = np.arange(start_frame_idx, i['num_frames'], frame_interval).astype(int)
                frame_indices = frame_indices[frame_indices < i['num_frames']]

                # comment out it to enable dynamic frames training
                if len(frame_indices) < self.num_frames and random.random() < self.drop_short_ratio:
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(len(frame_indices))
                    frame_indices = frame_indices[begin_index: end_index]
                    # frame_indices = frame_indices[:self.num_frames]  # head crop
                # to find a suitable end_frame_idx, to ensure we do not need pad video
                end_frame_idx = find_closest_y(len(frame_indices), vae_stride_t=4, model_ds_t=4)
                if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
                    cnt_too_short += 1
                    continue
                frame_indices = frame_indices[:end_frame_idx]

                i['sample_frame_index'] = frame_indices.tolist()
                new_cap_list.append(i)
                i['sample_num_frames'] = len(i['sample_frame_index'])  # will use in dataloader(group sampler)
                sample_num_frames.append(i['sample_num_frames'])
            elif path.endswith('.jpg'):  # image
                cnt_img += 1
                new_cap_list.append(i)
                i['sample_num_frames'] = 1
                sample_num_frames.append(i['sample_num_frames'])
            else:
                raise NameError(f"Unknown file extention {path.split('.')[-1]}, only support .mp4 for video and .jpg for image")
        # import ipdb;ipdb.set_trace()
        logger.info(f'no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short}, '
                f'no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, '
                f'Counter(sample_num_frames): {Counter(sample_num_frames)}, cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, '
                f'before filter: {len(cap_list)}, after filter: {len(new_cap_list)}')
        return new_cap_list, sample_num_frames
    
    def decord_read(self, path, predefine_num_frames):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        fps = decord_vr.get_avg_fps() if decord_vr.get_avg_fps() > 0 else 30.0
        # import ipdb;ipdb.set_trace()
        # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
        frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        start_frame_idx = 0  
        frame_indices = np.arange(start_frame_idx, total_frames, frame_interval).astype(int)
        frame_indices = frame_indices[frame_indices < total_frames]
        #import ipdb;ipdb.set_trace()
        # speed up
        max_speed_factor = len(frame_indices) / self.num_frames
        if self.speed_factor > 1 and max_speed_factor > 1:
            # speed_factor = random.uniform(1.0, min(self.speed_factor, max_speed_factor))
            speed_factor = min(self.speed_factor, max_speed_factor)
            target_frame_count = int(len(frame_indices) / speed_factor)
            speed_frame_idx = np.linspace(0, len(frame_indices) - 1, target_frame_count, dtype=int)
            frame_indices = frame_indices[speed_frame_idx]

        #  too long video will be temporal-crop randomly
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index: end_index]
            # frame_indices = frame_indices[:self.num_frames]  # head crop

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = find_closest_y(len(frame_indices), vae_stride_t=4, model_ds_t=4)
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(f'video ({path}) has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        frame_indices = frame_indices[:end_frame_idx]
        if predefine_num_frames != len(frame_indices):
            raise ValueError(f'predefine_num_frames ({predefine_num_frames}) is not equal with frame_indices ({len(frame_indices)})')
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(f'video ({path}) has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data):
        cap_lists = []
        with open(data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                sub_list = json.load(f)
            logger.info(f'Building {anno}...')
            for i in range(len(sub_list)):
                sub_list[i]['path'] = opj(folder, sub_list[i]['path'])
            cap_lists += sub_list
        return cap_lists

    # def get_img_cap_list(self):
    #     use_image_num = self.use_image_num if self.use_image_num != 0 else 1
    #     if npu_config is None:
    #         img_cap_lists = self.read_jsons(self.image_data, postfix=".jpg")
    #         img_cap_lists = [img_cap_lists[i: i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
    #     else:
    #         img_cap_lists = npu_config.try_load_pickle("img_cap_lists_all",
    #                                                    lambda: self.read_jsons(self.image_data, postfix=".jpg"))
    #         img_cap_lists = [img_cap_lists[i: i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
    #         img_cap_lists = img_cap_lists[npu_config.get_local_rank()::npu_config.N_NPU_PER_NODE]
    #     return img_cap_lists[:-1]  # drop last to avoid error length

    def get_cap_list(self):
        cap_lists = self.read_jsons(self.data)
        return cap_lists
