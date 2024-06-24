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

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
from PIL import Image

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


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
        self.vid_cap_list = []
        self.img_cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, img_cap_list, vid_cap_list, n_elements):
        self.num_workers = num_workers
        self.img_cap_list = img_cap_list
        self.vid_cap_list = vid_cap_list
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


class T2V_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.image_data = args.image_data
        self.video_data = args.video_data
        self.num_frames = args.num_frames
        self.use_image_num = args.use_image_num
        self.use_img_from_vid = args.use_img_from_vid
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.v_decoder = DecordInit()

        self.support_Chinese = True
        if not ('mt5' in args.text_encoder_name):
            self.support_Chinese = False

        if self.num_frames != 1:
            vid_cap_list = self.get_vid_cap_list()
            if self.use_image_num != 0 and not self.use_img_from_vid:
                img_cap_list = self.get_img_cap_list()
            else:
                img_cap_list = []
        else:
            img_cap_list = self.get_img_cap_list()
            vid_cap_list = []

        if self.num_frames != 1:
            n_elements = len(vid_cap_list)
        else:
            n_elements = len(img_cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, img_cap_list, vid_cap_list, n_elements)

        print(f"video length: {len(dataset_prog.vid_cap_list)}", flush=True)
        print(f"image length: {len(dataset_prog.img_cap_list)}", flush=True)

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
            video_data, image_data = {}, {}
            if self.num_frames != 1:
                video_data = self.get_video(idx)
                if self.use_image_num != 0:
                    if self.use_img_from_vid:
                        image_data = self.get_image_from_video(video_data)
                    else:
                        image_data = self.get_image(idx)
            else:
                image_data = self.get_image(idx)  # 1 frame video as image
            return dict(video_data=video_data, image_data=image_data)
        except Exception as e:
            # print(f'Error with {e}')
            # 打印异常堆栈
            if idx in dataset_prog.vid_cap_list:
                print(f"Caught an exception! {dataset_prog.vid_cap_list[idx]}")
            # traceback.print_exc()
            # traceback.print_stack()
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def get_video(self, idx):
        # npu_config.print_msg(f"current idx is {idx}")
        # video = random.choice([random_video_noise(65, 3, 336, 448), random_video_noise(65, 3, 1024, 1024), random_video_noise(65, 3, 360, 480)])
        # # print('random shape', video.shape)
        # input_ids = torch.ones(1, 120).to(torch.long).squeeze(0)
        # cond_mask = torch.cat([torch.ones(1, 60).to(torch.long), torch.ones(1, 60).to(torch.long)], dim=1).squeeze(0)

        video_path = dataset_prog.vid_cap_list[idx]['path']
        assert os.path.exists(video_path) and os.path.getsize(video_path) > 10240, f"file {video_path} has wrong size!"
        frame_idx = dataset_prog.vid_cap_list[idx]['frame_idx'] if hasattr(dataset_prog.vid_cap_list[idx], 'frame_idx') else None
        video = self.decord_read(video_path, frame_idx)

        h, w = video.shape[-2:]
        assert h / w <= 16 / 16 and h / w >= 4 / 16, f'Only videos with a ratio (h/w) less than 16/16 and more than 4/16 are supported. But found ratio is {round(h / w, 2)} with the shape of {video.shape}'
        t = video.shape[0]
        video = video[:(t - 1) // 4 * 4 + 1]
        video = self.transform(video)  # T C H W -> T C H W

        # video = torch.rand(221, 3, 480, 640)

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = dataset_prog.vid_cap_list[idx]['cap']

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
        return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)

    def get_image_from_video(self, video_data):
        select_image_idx = np.linspace(0, self.num_frames - 1, self.use_image_num, dtype=int)
        assert self.num_frames >= self.use_image_num
        image = [video_data['video'][:, i:i + 1] for i in select_image_idx]  # num_img [c, 1, h, w]
        input_ids = video_data['input_ids'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        cond_mask = video_data['cond_mask'].repeat(self.use_image_num, 1)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, idx):
        idx = idx % len(dataset_prog.img_cap_list)  # out of range
        image_data = dataset_prog.img_cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        image = [Image.open(i['path']).convert('RGB') for i in image_data]  # num_img [h, w, c]
        image = [torch.from_numpy(np.array(i)) for i in image]  # num_img [h, w, c]

        for i in image:
            assert not torch.any(torch.isnan(i)), 'before transform0'
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image]  # num_img [1 c h w]
        for i in image:
            assert not torch.any(torch.isnan(i)), 'before transform1'
        image = [self.transform(i) for i in image]  # num_img [1 C H W] -> num_img [1 C H W]

        for i in image:
            assert not torch.any(torch.isnan(i)), 'after transform'
        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = [i.transpose(0, 1) for i in image]  # num_img [1 C H W] -> num_img [C 1 H W]

        caps = [i['cap'] for i in image_data]
        text = [text_preprocessing(cap, support_Chinese=self.support_Chinese) for cap in caps]
        input_ids, cond_mask = [], []
        for t in text:
            t = t if random.random() > self.cfg else ""
            text_tokens_and_mask = self.tokenizer(
                t,
                max_length=self.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids.append(text_tokens_and_mask['input_ids'])
            cond_mask.append(text_tokens_and_mask['attention_mask'])
        input_ids = torch.cat(input_ids)  # self.use_image_num, l
        cond_mask = torch.cat(cond_mask)  # self.use_image_num, l
        return dict(image=image, input_ids=input_ids, cond_mask=cond_mask)

    def decord_read(self, path, frame_idx=None):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if frame_idx is None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        else:
            start_frame_ind, end_frame_ind = frame_idx.split(':')
            # start_frame_ind, end_frame_ind = int(start_frame_ind), int(end_frame_ind)
            start_frame_ind, end_frame_ind = int(start_frame_ind), int(start_frame_ind) + self.num_frames
        # assert end_frame_ind - start_frame_ind >= self.num_frames
        frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)
        # frame_indice = np.linspace(0, 63, self.num_frames, dtype=int)

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data, postfix=".jpg"):
        cap_lists = []
        with open(data, 'r') as f:
            folder_anno = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
        for folder, anno in folder_anno:
            with open(anno, 'r') as f:
                sub_list = json.load(f)
            print(f'Building {anno}...')
            for i in tqdm(range(len(sub_list))):
                sub_list[i]['path'] = opj(folder, sub_list[i]['path'])
            if npu_config is not None:
                sub_list = filter_json_by_existed_files(folder, sub_list, postfix=postfix)
            cap_lists += sub_list
        return cap_lists

    def get_img_cap_list(self):
        use_image_num = self.use_image_num if self.use_image_num != 0 else 1
        if npu_config is None:
            img_cap_lists = self.read_jsons(self.image_data, postfix=".jpg")
            img_cap_lists = [img_cap_lists[i: i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
        else:
            img_cap_lists = npu_config.try_load_pickle("img_cap_lists_only_mj_with_cn",
                                                       lambda: self.read_jsons(self.image_data, postfix=".jpg"))
            img_cap_lists = [img_cap_lists[i: i + use_image_num] for i in range(0, len(img_cap_lists), use_image_num)]
            img_cap_lists = img_cap_lists[npu_config.get_local_rank()::npu_config.N_NPU_PER_NODE]
        return img_cap_lists[:-1]  # drop last to avoid error length

    def get_vid_cap_list(self):
        if npu_config is None:
            vid_cap_lists = self.read_jsons(self.video_data, postfix=".mp4")
        else:
            vid_cap_lists = npu_config.try_load_pickle("vid_cap_lists5",
                                                       lambda: self.read_jsons(self.video_data, postfix=".mp4"))
            # npu_config.print_msg(f"length of vid_cap_lists is {len(vid_cap_lists)}")
            vid_cap_lists = vid_cap_lists[npu_config.get_local_rank()::npu_config.N_NPU_PER_NODE]
            vid_cap_lists_final = []
            for item in vid_cap_lists:
                if os.path.exists(item['path']) and os.path.getsize(item['path']) > 10240:
                    vid_cap_lists_final.append(item)
            vid_cap_lists = vid_cap_lists_final
            npu_config.print_msg(f"length of vid_cap_lists is {len(vid_cap_lists)}")

        return vid_cap_lists
