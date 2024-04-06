import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from timm.data import ImageDataset
from timm.data.readers import create_reader
from typing import Sequence
import math

# Using heap to store the remainder of token_num in each package
class heap_item:
    def __init__(self,remainder,idx):
        self.remainder=remainder
        self.idx=idx

    def __lt__(self, other):
        return self.remainder < other.remainder


# Using greedy algorithms to package images/videos into packages with token_num less than max_token_lim
def image_packaging(
    images,
    patch_size,
    max_token_lim = 1024,
    token_dropout_rate=0.0,
    version='i'
):

    packages = [[]]
    remainder_heap=[heap_item(-max_token_lim,0)]
    heapq.heapify(remainder_heap)
    for image in images:
        if version=='i':
            assert isinstance(image,torch.Tensor), 'image must be torch.Tensor'
            assert image.ndim == 3, 'image must be 3d tensor'
        elif version=='v':
            assert isinstance(image,torch.Tensor), 'video must be torch.Tensor'
            assert image.ndim == 4, 'video must be 4d tensor'
        else:
            raise ValueError('version must be i or v')
        image_h,image_w = image.shape[-2:]
        assert (image_w % patch_size) == 0 and (image_h % patch_size) == 0, f'image width and height must be divisible by patch size {patch_size}'
        patch_w,patch_h = image_w//patch_size,image_h//patch_size

        token_len = int(patch_w*patch_h*(1-token_dropout_rate))
        assert token_len <= max_token_lim, f'token length {token_len} exceeds max token length {max_token_lim}'

        package_idx=0
        while(package_idx<len(packages)):
            if remainder_heap[package_idx].remainder+token_len<=0:
                remainder_heap[package_idx].remainder+=token_len
                packages[package_idx].append(image)
                heapq.heapify(remainder_heap)
                break
            package_idx+=1
        if package_idx==len(packages):
            packages.append([image])
            heapq.heappush(remainder_heap,heap_item(-max_token_lim+token_len,package_idx))

    return packages

# Calculate token dropout rate, while setting default parameters according to the original paper
def cal_token_dropout_rate(step,start_step,temp,min=0.2,max=0.8):
    if step<start_step:
        return min
    weight=(step-start_step)/temp
    return min+(max-min)*1/(1+math.exp(-weight))

# Resize images to the shapes chosiest to their originals
class Resize_by_Patch(transforms.Resize):
    def forward(self, img):
        img_h,img_w = img.shape[-2:]
        if isinstance(self.size, int):
            if img_h%self.size==0:
                new_h=img_h
            else:
                new_h=(img_h//self.size+1)*self.size
            if img_w%self.size==0:
                new_w=img_w
            else:
                new_w=(img_w//self.size+1)*self.size
        if isinstance(self.size, Sequence):
            patch_h,patch_w=self.size
            if img_h%patch_h==0:
                new_h=img_h
            else:
                new_h=(img_h//patch_h+1)*patch_h
            if img_w%patch_w==0:
                new_w=img_w
            else:
                new_w=(img_w//patch_w+1)*patch_w
        return F.resize(img, [new_h,new_w], self.interpolation, self.max_size, self.antialias)

class ImgDataset(ImageDataset):
    def __init__(
        self,
        root,
        patch_size,
        reader=None,
        split='train',
        class_map=None,
        load_bytes=False,
        input_img_mode='RGB',
        target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            Resize_by_Patch(patch_size),
        ])
        self.target_transform = target_transform
        self._consecutive_errors = 0

# Load image tensors into sequence because their sizes are different
def my_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.stack(labels, dim=0)
    return images, labels