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
from decord import VideoReader
from os.path import join as opj
from collections import Counter

import jsonlines
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image
from accelerate.logging import get_logger

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from opensora.dataset.transform import get_params, longsideresize, add_masking_notice, motion_mapping_fun, calculate_statistics, \
    add_webvid_watermark_notice, clean_vidal, add_high_aesthetic_notice_image, add_aesthetic_notice_video, add_high_aesthetic_notice_image_human

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
        if torch_npu is not None:
            random.shuffle(self.elements)
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
        
def read_parquet(path):
    df = pd.read_parquet(path)
    data = df.to_dict(orient='records')
    return data

class T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.data = args.data
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.drop_short_ratio = args.drop_short_ratio
        self.hw_stride = args.hw_stride
        self.skip_low_resolution = args.skip_low_resolution
        self.force_resolution = args.force_resolution
        self.use_motion = args.use_motion
        assert self.speed_factor >= 1
        self.v_decoder = DecordInit()

        self.support_Chinese = True
        if not ('mt5' in args.text_encoder_name):
            self.support_Chinese = False

        s = time.time()

        # self.cache_pkl = f"/storage/dataset/{os.path.basename(self.data).replace('.txt', '_cache.pkl')}"
        # self.cache_json = f"/storage/dataset/{os.path.basename(self.data).replace('.txt', '_cache.json')}"
        
        # if os.path.exists(self.cache_pkl):
        #     print(f"Load cache from {self.cache_pkl}")
        #     with open(self.cache_pkl, 'rb') as f:
        #         load_data = pickle.load(f)
        #     cap_list, self.sample_size = load_data['cap_list'], load_data['sample_size']
        # if os.path.exists(self.cache_json):
        #     print(f"Load cache from {self.cache_json}")
        #     with open(self.cache_json, 'r') as f:
        #         load_data = json.load(f)
        #     cap_list, self.sample_size = load_data['cap_list'], load_data['sample_size']
        # else:
        # data_root, cap_list = self.get_cap_list()
        # assert len(cap_list) > 0
        cap_list, self.sample_size, _ = self.define_frame_index(self.data)
        # save_data = dict(cap_list=cap_list, sample_size=self.sample_size)
        # print(f"Save cache to {self.cache_pkl}")
        # with open(self.cache_pkl, 'wb') as f:
        #     pickle.dump(save_data, f)
        # print(f"Save cache to {self.cache_json}")
        # with open(self.cache_json, 'w') as f:
        #     json.dump(save_data, f)
        e = time.time()
        print('time', e-s)
        self.lengths = self.sample_size

        n_elements = len(cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list, n_elements)
        print(f"data length: {len(dataset_prog.cap_list)}")

    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return dataset_prog.n_elements

    def __getitem__(self, idx):
        if npu_config is not None:
            worker_info = get_worker_info()
            idx = dataset_prog.get_item(worker_info)
        try:
            # print('idx:', idx)
            data = self.get_data(idx)
            return data
        except Exception as e:
            print(e)
            logger.info(f'Error with {e}')
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

        video_data = dataset_prog.cap_list[idx]
        video_path = video_data['path']
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indice = dataset_prog.cap_list[idx]['sample_frame_index']
        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']
        video = self.decord_read(video_path, predefine_frame_indice=frame_indice)
        # import ipdb;ipdb.set_trace()
        video = self.transform(video)  # T C H W -> T C H W
        assert video.shape[2] == sample_h and video.shape[3] == sample_w

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = video_data['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if '/VIDAL-10M/' in video_path:
            text = [clean_vidal(text[0])]
        if '/Webvid-10M/' in video_path:
            text = [add_webvid_watermark_notice(text[0])]
        if not (video_data.get('aesthetic', None) is None):
            text = [add_aesthetic_notice_video(text[0], video_data['aesthetic'])]

        text = [text[0].replace(' image ', ' video ').replace(' image,', ' video,')]
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
        if self.use_motion:
            motion_score = motion_mapping_fun(video_data['motion_score'])
            return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask, motion_score=motion_score)
        else:
            return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask, motion_score=None)

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        sample_h = image_data['resolution']['sample_height']
        sample_w = image_data['resolution']['sample_width']

        # import ipdb;ipdb.set_trace()
        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]

        # import ipdb;ipdb.set_trace()
        image = self.transform(image) #  [1 C H W] -> num_img [1 C H W]
        assert image.shape[2] == sample_h, image.shape[3] == sample_w
        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = image_data['cap'] if isinstance(image_data['cap'], list) else [image_data['cap']]
        caps = [random.choice(caps)]
        if '/sam/' in image_data['path']:
            caps = [add_masking_notice(caps[0])]
        if 'ideogram' in image_data['path']:
            caps = [add_high_aesthetic_notice_image(caps[0])]
        if 'human_images' in image_data['path']:
            caps = [add_high_aesthetic_notice_image_human(caps[0])]
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
        if self.use_motion:
            motion_score = motion_mapping_fun(image_data['motion_score'])
            return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask, motion_score=motion_score)
        else:
            return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask, motion_score=None)

    def define_frame_index(self, data):
        
        new_cap_list = []
        sample_size = []
        motion_score = []
        aesthetic_score = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_no_motion = 0
        cnt_no_aesthetic = 0
        cnt_resolution_mismatch = 0
        cnt_movie = 0
        cnt_img = 0
        cnt = 0

        with open(data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for sub_root, anno in tqdm(folder_anno):
            logger.info(f'Building {anno}...')
            if anno.endswith('.json'):
                with open(anno, 'r') as f:
                    sub_list = json.load(f)
            elif anno.endswith('.pkl'):
                with open(anno, 'rb') as f:
                    sub_list = pickle.load(f)
            elif anno.endswith('.parquet'):
                sub_list = read_parquet(anno)
            else:
                raise NotImplementedError
            # with jsonlines.open(anno) as sub_list:
            for i in tqdm(sub_list):
                cnt += 1
                path = os.path.join(sub_root, i['path'])
                i['path'] = path
                cap = i.get('cap', None)

                if i.get('aesthetic', None) is None:
                    cnt_no_aesthetic += 1
                else:
                    aesthetic_score.append(i['aesthetic'])

                # ======no caption=====
                if cap is None:
                    cnt_no_cap += 1
                    continue
                # ======no motion=====
                if self.use_motion:
                    if '.mp4' in path and i.get('motion_average', None) is None and i.get('motion', None) is None:
                        cnt_no_motion += 1
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
                            tr_h, tr_w = longsideresize(height, width, (self.max_height, self.max_width), self.skip_low_resolution)
                            _, _, sample_h, sample_w = get_params(tr_h, tr_w, self.hw_stride)
                            if sample_h <= 0 or sample_w <= 0:
                                cnt_resolution_mismatch += 1
                                continue
                            i['resolution'].update(dict(sample_height=sample_h, sample_width=sample_w))
                        else:
                            aspect = self.max_height / self.max_width
                            hw_aspect_thr = 1.85
                            is_pick = filter_resolution(height, width, max_h_div_w_ratio=hw_aspect_thr*aspect, 
                                                        min_h_div_w_ratio=1/hw_aspect_thr*aspect)
                            if not is_pick:
                                cnt_resolution_mismatch += 1
                                continue
                            sample_h, sample_w = self.max_height, self.max_width
                            i['resolution'].update(dict(sample_height=sample_h, sample_width=sample_w))


                if path.endswith('.mp4'):
                    # ======no fps and duration=====
                    duration = i.get('duration', None)
                    fps = i.get('fps', None)
                    if fps is None or duration is None:
                        continue

                    i['num_frames'] = int(fps * duration)
                    # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration. 
                    if i['num_frames'] > 6.0 * (self.num_frames * fps / self.train_fps * self.speed_factor):  # too long video is not suitable for this training stage (self.num_frames)
                        cnt_too_long += 1
                        continue
                    # if i['num_frames'] < 1.0/1 * (self.num_frames * fps / self.train_fps * self.speed_factor):  # too short video is not suitable for this training stage
                    #     cnt_too_short += 1
                    #     continue 

                    # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                    frame_interval = fps / self.train_fps
                    start_frame_idx = 10 if '/storage/dataset/movie' in i['path'] else 0  # special video
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

                    if '/storage/dataset/movie' in i['path']:
                        cnt_movie += 1
                        
                    i['sample_frame_index'] = frame_indices.tolist()
                    i['motion_score'] = i.get('motion_average', None) or i.get('motion')

                    new_cap_list.append(i)
                    # i['sample_num_frames'] = len(i['sample_frame_index'])  # will use in dataloader(group sampler)


                elif path.endswith('.jpg'):  # image
                    cnt_img += 1
                    i['sample_frame_index'] = [0]
                    i['motion_score'] = 1.0
                    new_cap_list.append(i)
                    # i['sample_num_frames'] = len(i['sample_frame_index'])  # will use in dataloader(group sampler)
                
                else:
                    raise NameError(f"Unknown file extention {path.split('.')[-1]}, only support .mp4 for video and .jpg for image")

                sample_size.append(f"{len(i['sample_frame_index'])}x{sample_h}x{sample_w}")
                if self.use_motion:
                    motion_score.append(i['motion_score'])

        logger.info(f'no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short}, '
                f'no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, '
                f'Counter(sample_size): {Counter(sample_size)}, cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, '
                f'before filter: {cnt}, after filter: {len(new_cap_list)}')
        if self.use_motion:
            stats_motion = calculate_statistics(motion_score)
            logger.info(f"before filter: {cnt}, after filter: {len(new_cap_list)} | "
                        f"motion_score: {len(motion_score)}, cnt_no_motion: {cnt_no_motion} | "
                        f"{len([i for i in motion_score if i>=0.95])} > 0.95, 0.7 > {len([i for i in motion_score if i<=0.7])} "
                        f"Mean: {stats_motion['mean']}, Var: {stats_motion['variance']}, Std: {stats_motion['std_dev']}, "
                        f"Min: {stats_motion['min']}, Max: {stats_motion['max']}")
        
        if len(aesthetic_score) > 0:
            stats_aesthetic = calculate_statistics(aesthetic_score)
            logger.info(f"before filter: {cnt}, after filter: {len(new_cap_list)} | "
                        f"aesthetic_score: {len(aesthetic_score)}, cnt_no_aesthetic: {cnt_no_aesthetic} | "
                        f"{len([i for i in aesthetic_score if i>=5.75])} > 5.75, 4.5 > {len([i for i in aesthetic_score if i<=4.5])} "
                        f"Mean: {stats_aesthetic['mean']}, Var: {stats_aesthetic['variance']}, Std: {stats_aesthetic['std_dev']}, "
                        f"Min: {stats_aesthetic['min']}, Max: {stats_aesthetic['max']}")
            
        # import ipdb;ipdb.set_trace()

        return new_cap_list, sample_size, motion_score
    
    def decord_read(self, path, predefine_frame_indice):
        predefine_num_frames = len(predefine_frame_indice)
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        fps = decord_vr.get_avg_fps() if decord_vr.get_avg_fps() > 0 else 24.0
        
        # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
        frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        start_frame_idx = 10 if '/storage/dataset/movie' in path else 0  # special video
        frame_indices = np.arange(start_frame_idx, total_frames, frame_interval).astype(int)
        frame_indices = frame_indices[frame_indices < total_frames]
        
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
            raise ValueError(f'video ({path}) predefine_num_frames ({predefine_num_frames}) ({predefine_frame_indice}) is not equal with frame_indices ({len(frame_indices)}) ({frame_indices})')
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(f'video ({path}) has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})')
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data, postfix=".jpg"):
        data_roots = []
        cap_lists = []
        with open(data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in tqdm(folder_anno):
            logger.info(f'Building {anno}...')
            if anno.endswith('.json'):
                with open(anno, 'r') as f:
                    sub_list = json.load(f)
            elif anno.endswith('.pkl'):
                with open(anno, 'rb') as f:
                    sub_list = pickle.load(f)
            elif anno.endswith('.parquet'):
                sub_list = read_parquet(anno)
            else:
                raise NotImplementedError
            # for i in tqdm(range(len(sub_list))):
            #     sub_list[i]['path'] = opj(folder, sub_list[i]['path'])
            # if npu_config is not None:
            #     if "civitai" in anno or "ideogram" in anno or "human" in anno:
            #         sub_list = sub_list[npu_config.get_node_id()::npu_config.get_node_size()]
            #     else:
            #         sub_list = filter_json_by_existed_files(folder, sub_list, postfix=postfix)
            data_roots.append(folder)
            cap_lists.append(sub_list)
        return data_roots, cap_lists

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
        if npu_config is None:
            data_roots, cap_lists = self.read_jsons(self.data, postfix=".mp4")
        else:
            raise NotImplementedError('need return data_roots')
            cap_lists = npu_config.try_load_pickle("cap_lists5",
                                                       lambda: self.read_jsons(self.data, postfix=".mp4"))
            # npu_config.print_msg(f"length of cap_lists is {len(cap_lists)}")
            cap_lists = cap_lists[npu_config.get_local_rank()::npu_config.N_NPU_PER_NODE]
            cap_lists_final = []
            for item in cap_lists:
                if os.path.exists(item['path']) and os.path.getsize(item['path']) > 10240:
                    cap_lists_final.append(item)
            cap_lists = cap_lists_final
            npu_config.print_msg(f"length of cap_lists is {len(cap_lists)}")

        return data_roots, cap_lists
