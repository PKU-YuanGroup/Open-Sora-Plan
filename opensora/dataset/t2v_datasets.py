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
import pickle
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from os.path import join as opj
from collections import Counter

import cv2
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger
import gc

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from opensora.dataset.transform import get_params, maxhwresize, add_masking_notice, calculate_statistics, \
    add_aesthetic_notice_image, add_aesthetic_notice_video

import decord
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logger = get_logger(__name__)

def filter_json_by_existed_files(directory, data, postfix=".mp4"):
    # 构建搜索模式，以匹配指定后缀的文件
    pattern = os.path.join(directory, '**', f'*{postfix}')
    mp4_files = glob.glob(pattern, recursive=True)  # 使用glob查找所有匹配的文件

    # 使用文件的绝对路径构建集合
    mp4_files_set = set(os.path.abspath(path) for path in mp4_files)

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
        
        print(f"n_elements: {len(self.elements)}", flush=True)
        # if torch_npu is not None:
        #     random.shuffle(self.elements)
        #     for i in range(self.num_workers):
        #         self.n_used_elements[i] = 0
        #         per_worker = int(math.ceil(len(self.elements) / float(self.num_workers)))
        #         start = i * per_worker
        #         end = min(start + per_worker, len(self.elements))
        #         self.worker_elements[i] = self.elements[start: end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][self.n_used_elements[worker_id] % len(self.worker_elements[worker_id])]
        self.n_used_elements[worker_id] += 1
        return idx


dataset_prog = DataSetProg()

def find_closest_y(x, vae_stride_t=4, model_ds_t=1):
    min_num_frames = 29
    if x < min_num_frames:
        return -1  
    for y in range(x, min_num_frames - 1, -1):
        if (y - 1) % vae_stride_t == 0 and ((y - 1) // vae_stride_t + 1) % model_ds_t == 0:
            # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
            # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
            # 8, 1: y in [33, 41, 49, 57, 65, 73, 81, 89, 97, 105]
            # 8, 2: y in [41, 57, 73, 89, 105]
            # 8, 4: y in [57, 89]
            # 8, 8: y in [57]
            return y
    return -1 

def filter_resolution(h, w, max_h_div_w_ratio=17/16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False
        
def read_parquet(path):
    df = pd.read_parquet(path)
    data = df.to_dict(orient='records')
    return data



class DecordDecoder(object):
    def __init__(self, url, num_threads=1):

        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.reader = decord.VideoReader(url,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)

    def get_avg_fps(self):
        return self.reader.get_avg_fps() if self.reader.get_avg_fps() > 0 else 30.0

    def get_num_frames(self):
        return len(self.reader)

    def get_height(self):
        return self.reader[0].shape[0] if self.get_num_frames() > 0 else 0

    def get_width(self):
        return self.reader[0].shape[1] if self.get_num_frames() > 0 else 0

    # output shape [T, H, W, C]
    def get_batch(self, frame_indices):
        try:
            #frame_indices[0] = 1000
            video_data = self.reader.get_batch(frame_indices).asnumpy()
            video_data = torch.from_numpy(video_data)
            return video_data
        except Exception as e:
            print('get_batch execption:', e)
            return None
        
class T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer_1, tokenizer_2):
        self.data = args.data
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.drop_short_ratio = args.drop_short_ratio
        self.hw_stride = args.hw_stride
        self.force_resolution = args.force_resolution
        self.max_hxw = args.max_hxw
        self.min_hxw = args.min_hxw
        self.sp_size = args.sp_size
        assert self.speed_factor >= 1
        self.video_reader = 'decord' if args.use_decord else 'opencv'
        self.ae_stride_t = args.ae_stride_t
        self.total_batch_size = args.total_batch_size
        self.seed = 42
        self.generator = torch.Generator().manual_seed(self.seed) 
        self.hw_aspect_thr = 2.0  # just a threshold
        self.too_long_factor = 5.0

        self.support_Chinese = False
        if 'mt5' in args.text_encoder_name_1:
            self.support_Chinese = True
        if args.text_encoder_name_2 is not None and 'mt5' in args.text_encoder_name_2:
            self.support_Chinese = True

        s = time.time()
        cap_list, self.sample_size, self.shape_idx_dict = self.define_frame_index(self.data)
        e = time.time()
        print(f'Build data time: {e-s}')
        self.lengths = self.sample_size

        n_elements = len(cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list, n_elements)
        print(f"Data length: {len(dataset_prog.cap_list)}")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.timeout = 60

    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return dataset_prog.n_elements

    def __getitem__(self, idx):
        try:
            future = self.executor.submit(self.get_data, idx)
            data = future.result(timeout=self.timeout) 
            # data = self.get_data(idx)
            return data
        except Exception as e:
            if len(str(e)) < 2:
                e = f"TimeoutError, {self.timeout}s timeout occur with {dataset_prog.cap_list[idx]['path']}"
            print(f'Error with {e}')
            index_cand = self.shape_idx_dict[self.sample_size[idx]]  # pick same shape
            return self.__getitem__(random.choice(index_cand))

    def get_data(self, idx):
        path = dataset_prog.cap_list[idx]['path']
        if path.endswith('.mp4'):
            return self.get_video(idx)
        else:
            return self.get_image(idx)
    
    def get_video(self, idx):
        video_data = dataset_prog.cap_list[idx]
        video_path = video_data['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']
        if self.video_reader == 'decord':
            video = self.decord_read(video_data)
        elif self.video_reader == 'opencv':
            video = self.opencv_read(video_data)
        else:
            NotImplementedError(f'Found {self.video_reader}, but support decord or opencv')
        # import ipdb;ipdb.set_trace()
        video = self.transform(video)  # T C H W -> T C H W
        assert video.shape[2] == sample_h and video.shape[3] == sample_w, f'sample_h ({sample_h}), sample_w ({sample_w}), video ({video.shape})'

        # video = torch.rand(105, 3, 640, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = video_data['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if video_data.get('aesthetic', None) is not None or video_data.get('aes', None) is not None:
            aes = video_data.get('aesthetic', None) or video_data.get('aes', None)
            text = [add_aesthetic_notice_video(text[0], aes)]
        text = text_preprocessing(text, support_Chinese=self.support_Chinese)

        text = text if random.random() > self.cfg else ""

        text_tokens_and_mask_1 = self.tokenizer_1(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_1 = text_tokens_and_mask_1['input_ids']
        cond_mask_1 = text_tokens_and_mask_1['attention_mask']
        
        input_ids_2, cond_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids_2 = text_tokens_and_mask_2['input_ids']
            cond_mask_2 = text_tokens_and_mask_2['attention_mask']

        return dict(
            pixel_values=video, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, 
            input_ids_2=input_ids_2, cond_mask_2=cond_mask_2,
            )

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        sample_h = image_data['resolution']['sample_height']
        sample_w = image_data['resolution']['sample_width']

        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]

        image = self.transform(image) #  [1 C H W] -> num_img [1 C H W]
        assert image.shape[2] == sample_h and image.shape[3] == sample_w, f"image_data: {image_data}, but found image {image.shape}"
        # image = torch.rand(1, 3, sample_h, sample_w)
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = image_data['cap'] if isinstance(image_data['cap'], list) else [image_data['cap']]
        caps = [random.choice(caps)]
        if image_data.get('aesthetic', None) is not None or image_data.get('aes', None) is not None:
            aes = image_data.get('aesthetic', None) or image_data.get('aes', None)
            caps = [add_aesthetic_notice_image(caps[0], aes)]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        text = text if random.random() > self.cfg else ""

        text_tokens_and_mask_1 = self.tokenizer_1(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_1 = text_tokens_and_mask_1['input_ids']  # 1, l
        cond_mask_1 = text_tokens_and_mask_1['attention_mask']  # 1, l
        
        input_ids_2, cond_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids_2 = text_tokens_and_mask_2['input_ids']  # 1, l
            cond_mask_2 = text_tokens_and_mask_2['attention_mask']  # 1, l

        return dict(
            pixel_values=image, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, 
            input_ids_2=input_ids_2, cond_mask_2=cond_mask_2
            )

    def define_frame_index(self, data):
        
        shape_idx_dict = {}
        new_cap_list = []
        sample_size = []
        aesthetic_score = []
        cnt_vid = 0
        cnt_img = 0
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_no_aesthetic = 0
        cnt_img_res_mismatch_stride = 0
        cnt_vid_res_mismatch_stride = 0
        cnt_img_aspect_mismatch = 0
        cnt_vid_aspect_mismatch = 0
        cnt_img_res_too_small = 0
        cnt_vid_res_too_small = 0
        cnt_vid_after_filter = 0
        cnt_img_after_filter = 0
        cnt = 0
        

        with open(data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for sub_root, anno in tqdm(folder_anno):
            print(f'Building {anno}...')
            if anno.endswith('.json'):
                with open(anno, 'r') as f:
                    sub_list = json.load(f)
            elif anno.endswith('.pkl'):
                with open(anno, "rb") as f: 
                    sub_list = pickle.load(f)
            for index, i in enumerate(tqdm(sub_list)):
                cnt += 1
                path = os.path.join(sub_root, i['path'])
                i['path'] = path
                if path.endswith('.mp4'):
                    cnt_vid += 1
                elif path.endswith('.jpg'):
                    cnt_img += 1

                # ======no aesthetic=====
                if i.get('aesthetic', None) is None or i.get('aes', None) is None:
                    cnt_no_aesthetic += 1
                else:
                    aesthetic_score.append(i.get('aesthetic', None) or i.get('aes', None))

                # ======no caption=====
                cap = i.get('cap', None)
                if cap is None:
                    cnt_no_cap += 1
                    continue

                # ======resolution mismatch=====
                if i.get('resolution', None) is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    if i['resolution'].get('height', None) is None or i['resolution'].get('width', None) is None:
                        cnt_no_resolution += 1
                        continue
                    else:
                        height, width = i['resolution']['height'], i['resolution']['width']
                        if not self.force_resolution:
                            if height <= 0 or width <= 0:
                                cnt_no_resolution += 1
                                continue
                            
                            tr_h, tr_w = maxhwresize(height, width, self.max_hxw)
                            _, _, sample_h, sample_w = get_params(tr_h, tr_w, self.hw_stride)

                            if sample_h <= 0 or sample_w <= 0:
                                if path.endswith('.mp4'):
                                    cnt_vid_res_mismatch_stride += 1
                                elif path.endswith('.jpg'):
                                    cnt_img_res_mismatch_stride += 1
                                continue
                            
                            # filter min_hxw
                            if sample_h * sample_w < self.min_hxw:
                                if path.endswith('.mp4'):
                                    cnt_vid_res_too_small += 1
                                elif path.endswith('.jpg'):
                                    cnt_img_res_too_small += 1
                                continue

                            # filter aspect
                            is_pick = filter_resolution(
                                sample_h, sample_w, max_h_div_w_ratio=self.hw_aspect_thr, min_h_div_w_ratio=1/self.hw_aspect_thr
                                )
                            if not is_pick:
                                if path.endswith('.mp4'):
                                    cnt_vid_aspect_mismatch += 1
                                elif path.endswith('.jpg'):
                                    cnt_img_aspect_mismatch += 1
                                continue

                            i['resolution'].update(dict(sample_height=sample_h, sample_width=sample_w))
                            
                        else:
                            aspect = self.max_height / self.max_width
                            is_pick = filter_resolution(
                                height, width, max_h_div_w_ratio=self.hw_aspect_thr*aspect, min_h_div_w_ratio=1/self.hw_aspect_thr*aspect
                                )
                            if not is_pick:
                                if path.endswith('.mp4'):
                                    cnt_vid_aspect_mismatch += 1
                                elif path.endswith('.jpg'):
                                    cnt_img_aspect_mismatch += 1
                                continue
                            sample_h, sample_w = self.max_height, self.max_width
                            
                            i['resolution'].update(dict(sample_height=sample_h, sample_width=sample_w))


                if path.endswith('.mp4'):
                    fps = i.get('fps', 24)
                    # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration. 
                    if i['num_frames'] > self.too_long_factor * (self.num_frames * fps / self.train_fps * self.speed_factor):  # too long video is not suitable for this training stage (self.num_frames)
                        cnt_too_long += 1
                        continue

                    # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                    frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
                    start_frame_idx = i.get('cut', [0])[0]
                    i['start_frame_idx'] = start_frame_idx
                    frame_indices = np.arange(start_frame_idx, start_frame_idx+i['num_frames'], frame_interval).astype(int)
                    frame_indices = frame_indices[frame_indices < start_frame_idx+i['num_frames']]

                    # comment out it to enable dynamic frames training
                    if len(frame_indices) < self.num_frames and torch.rand(1, generator=self.generator).item() < self.drop_short_ratio:
                        cnt_too_short += 1
                        continue

                    #  too long video will be temporal-crop randomly
                    if len(frame_indices) > self.num_frames:
                        begin_index, end_index = self.temporal_sample(len(frame_indices))
                        frame_indices = frame_indices[begin_index: end_index]
                        # frame_indices = frame_indices[:self.num_frames]  # head crop
                    # to find a suitable end_frame_idx, to ensure we do not need pad video
                    end_frame_idx = find_closest_y(
                        len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
                        )
                    if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
                        cnt_too_short += 1
                        continue
                    frame_indices = frame_indices[:end_frame_idx]

                    i['sample_frame_index'] = frame_indices.tolist()

                    new_cap_list.append(i)
                    cnt_vid_after_filter += 1

                elif path.endswith('.jpg'):  # image
                    cnt_img_after_filter += 1
                    i['sample_frame_index'] = [0]
                    new_cap_list.append(i)
                
                else:
                    raise NameError(f"Unknown file extention {path.split('.')[-1]}, only support .mp4 for video and .jpg for image")

                pre_define_shape = f"{len(i['sample_frame_index'])}x{sample_h}x{sample_w}"
                sample_size.append(pre_define_shape)
                # if shape_idx_dict.get(pre_define_shape, None) is None:
                #     shape_idx_dict[pre_define_shape] = [index]
                # else:
                #     shape_idx_dict[pre_define_shape].append(index)
        counter = Counter(sample_size)
        counter_cp = counter
        if not self.force_resolution and self.max_hxw is not None and self.min_hxw is not None:
            assert all([np.prod(np.array(k.split('x')[1:]).astype(np.int32)) <= self.max_hxw for k in counter_cp.keys()])
            assert all([np.prod(np.array(k.split('x')[1:]).astype(np.int32)) >= self.min_hxw for k in counter_cp.keys()])

        len_before_filter_major = len(sample_size)
        filter_major_num = 4 * self.total_batch_size
        new_cap_list, sample_size = zip(*[[i, j] for i, j in zip(new_cap_list, sample_size) if counter[j] >= filter_major_num])
        for idx, shape in enumerate(sample_size):
            if shape_idx_dict.get(shape, None) is None:
                shape_idx_dict[shape] = [idx]
            else:
                shape_idx_dict[shape].append(idx)
        cnt_filter_minority = len_before_filter_major - len(sample_size) 
        counter = Counter(sample_size)
        
        print(f'no_cap: {cnt_no_cap}, no_resolution: {cnt_no_resolution}\n'
                f'too_long: {cnt_too_long}, too_short: {cnt_too_short}\n'
                f'cnt_img_res_mismatch_stride: {cnt_img_res_mismatch_stride}, cnt_vid_res_mismatch_stride: {cnt_vid_res_mismatch_stride}\n'
                f'cnt_img_res_too_small: {cnt_img_res_too_small}, cnt_vid_res_too_small: {cnt_vid_res_too_small}\n'
                f'cnt_img_aspect_mismatch: {cnt_img_aspect_mismatch}, cnt_vid_aspect_mismatch: {cnt_vid_aspect_mismatch}\n'
                f'cnt_filter_minority: {cnt_filter_minority}\n'
                f'Counter(sample_size): {counter}\n'
                f'cnt_vid: {cnt_vid}, cnt_vid_after_filter: {cnt_vid_after_filter}, use_ratio: {round(cnt_vid_after_filter/(cnt_vid+1e-6), 5)*100}%\n'
                f'cnt_img: {cnt_img}, cnt_img_after_filter: {cnt_img_after_filter}, use_ratio: {round(cnt_img_after_filter/(cnt_img+1e-6), 5)*100}%\n'
                f'before filter: {cnt}, after filter: {len(new_cap_list)}, use_ratio: {round(len(new_cap_list)/cnt, 5)*100}%')
        # import ipdb;ipdb.set_trace()
        
        if len(aesthetic_score) > 0:
            stats_aesthetic = calculate_statistics(aesthetic_score)
            print(f"before filter: {cnt}, after filter: {len(new_cap_list)}\n"
                f"aesthetic_score: {len(aesthetic_score)}, cnt_no_aesthetic: {cnt_no_aesthetic}\n"
                f"{len([i for i in aesthetic_score if i>=5.75])} > 5.75, 4.5 > {len([i for i in aesthetic_score if i<=4.5])}\n"
                f"Mean: {stats_aesthetic['mean']}, Var: {stats_aesthetic['variance']}, Std: {stats_aesthetic['std_dev']}\n"
                f"Min: {stats_aesthetic['min']}, Max: {stats_aesthetic['max']}")

        return new_cap_list, sample_size, shape_idx_dict
    
    def decord_read(self, video_data):
        path = video_data['path']
        predefine_frame_indice = video_data['sample_frame_index']
        start_frame_idx = video_data['start_frame_idx']
        clip_total_frames = video_data['num_frames']
        fps = video_data['fps']
        s_x, e_x, s_y, e_y = video_data.get('crop', [None, None, None, None])

        predefine_num_frames = len(predefine_frame_indice)
        # decord_vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        decord_vr = DecordDecoder(path)

        frame_indices = self.get_actual_frame(
            fps, start_frame_idx, clip_total_frames, path, predefine_num_frames, predefine_frame_indice
            )
        
        # video_data = decord_vr.get_batch(frame_indices).asnumpy()
        # video_data = torch.from_numpy(video_data)
        video_data = decord_vr.get_batch(frame_indices)
        if video_data is not None:
            video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
            if s_y is not None:
                video_data = video_data[:, :, s_y: e_y, s_x: e_x]
        else:
            raise ValueError(f'Get video_data {video_data}')
        # del decord_vr
        # gc.collect()
        return video_data
    
    def opencv_read(self, video_data):
        path = video_data['path']
        predefine_frame_indice = video_data['sample_frame_index']
        start_frame_idx = video_data['start_frame_idx']
        clip_total_frames = video_data['num_frames']
        fps = video_data['fps']
        s_x, e_x, s_y, e_y = video_data.get('crop', [None, None, None, None])

        predefine_num_frames = len(predefine_frame_indice)
        cv2_vr = cv2.VideoCapture(path)
        if not cv2_vr.isOpened():
            raise ValueError(f'can not open {path}')
        frame_indices = self.get_actual_frame(
            fps, start_frame_idx, clip_total_frames, path, predefine_num_frames, predefine_frame_indice
            )

        video_data = []
        for frame_idx in frame_indices:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data)  # (T C H W)
        if s_y is not None:
            video_data = video_data[:, :, s_y: e_y, s_x: e_x]
        return video_data

    def get_actual_frame(self, fps, start_frame_idx, clip_total_frames, path, predefine_num_frames, predefine_frame_indice):
        # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
        frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        frame_indices = np.arange(start_frame_idx, start_frame_idx+clip_total_frames, frame_interval).astype(int)
        frame_indices = frame_indices[frame_indices < start_frame_idx+clip_total_frames]
        
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
        end_frame_idx = find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
            )
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(f'video ({path}) has {clip_total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        frame_indices = frame_indices[:end_frame_idx]
        if predefine_num_frames != len(frame_indices):
            raise ValueError(f'video ({path}) predefine_num_frames ({predefine_num_frames}) ({predefine_frame_indice}) is not equal with frame_indices ({len(frame_indices)}) ({frame_indices})')
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(f'video ({path}) has {clip_total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        return frame_indices
