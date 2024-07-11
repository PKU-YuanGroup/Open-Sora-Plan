
from torch.utils.data import Dataset

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

from .t2v_datasets import filter_json_by_existed_files, random_video_noise, find_closest_y, filter_resolution
from .t2v_datasets import SingletonMeta, DataSetProg
from .t2v_datasets import T2V_dataset

logger = get_logger(__name__)


dataset_prog = DataSetProg()

class VideoIP_dataset(T2V_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        super().__init__(args, transform, temporal_sample, tokenizer, transform_topcrop)

        if self.num_frames != 1:
            # inpaint
            # The proportion of executing the i2v task.
            self.i2v_ratio = args.i2v_ratio
            self.transition_ratio = args.transition_ratio
            self.clear_video_ratio = args.clear_video_ratio
            self.default_text_ratio = args.default_text_ratio
            assert self.i2v_ratio + self.transition_ratio + self.clear_video_ratio < 1, 'The sum of i2v_ratio, transition_ratio and clear video ratio should be less than 1.'
