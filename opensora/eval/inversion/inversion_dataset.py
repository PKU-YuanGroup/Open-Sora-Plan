import os

from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, ToTensor, Compose, Resize, Normalize
from PIL import Image
import pickle as pkl
import json
import random
from opensora.dataset.transform import CenterCropResizeVideo, ToTensorVideo
from opensora.models.causalvideovae import ae_norm, ae_denorm

import torch
import numpy as np
from einops import rearrange

class InversionValidImageDataset(Dataset):
    """
    For image valid.
    """

    def __init__(self, args) -> None:
        self.dataset = []
        norm_fun = ae_norm[args.ae]
        self.no_condition = args.no_condition
        self.transform = Compose(
            [
                ToTensorVideo(),
                CenterCropResizeVideo((args.height, args.width)),
                norm_fun,  # output [-1, 1]
            ]
        )
    
        self.dataset = self._load_dataset(args.data_path, args.data_root, args.save_img_path)
        if args.num_samples > 0:
            self.dataset = self.dataset[:args.num_samples]
        print(f'exist {len(self.dataset)}')

    def _get_cap_or_label(self, item):
        if self.no_condition:
            return ""
        if len(item['caption']) > 0:
            return item['caption'][0]
        if len(item['label']) > 0:
            return item['label'][0]
        return ""
    def _load_dataset(self, data_path, data_root, save_root):
        with open(data_path, "r") as f:
            sub_data = json.load(f)
        print(f'total {len(sub_data)}')
        dataset = [
            (
                os.path.join(data_root, line["path"]),
                self._get_cap_or_label(line),
                line["path"], 
            )
            for line in sub_data if not os.path.exists(os.path.join(save_root, line["path"]).replace('.jpg', '_gt.png'))
        ]
        return dataset
    def __getitem__(self, index):
        image_path, caption, rela_path = self.dataset[index]
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]
        image = self.transform(image)
        
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]
        return {
            "image": image,
            "caption": caption, 
            "rela_path": rela_path
        }

    def __len__(self):
        return len(self.dataset)
    



class InversionEvalImageDataset(Dataset):
    """
    For image valid.
    """

    def __init__(self, args) -> None:
        with open(args.data_file, "r") as f:
            data = json.load(f)
            
        dataset = [os.path.join(args.data_root, i["path"].replace('.jpg', '_gt.png')) for i in data]
        dataset = [i.replace('_gt.png', '') for i in dataset if os.path.exists(i)]

        self.dataset = [
            [f'{i}_gt.png'] + [f'{i}_{inverse_steps}.png' for inverse_steps in args.num_inverse_steps]
            for i in dataset
        ]
        if args.num_samples > 0:
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:args.num_samples]

    def __getitem__(self, index):
        image_paths = self.dataset[index]
        image = [torch.from_numpy(np.array(Image.open(i).convert("RGB")))/255.0 for i in image_paths]
        image = [rearrange(i, 'h w c -> c h w').unsqueeze(0) for i in image]  #  [1 c h w]
        image = torch.stack(image)
        return image

    def __len__(self):
        return len(self.dataset)
