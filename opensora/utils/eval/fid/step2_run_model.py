import os
import torch
import argparse
import pandas as pd
from opensora.utils.utils import set_seed
from copy import deepcopy
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
import torch
from glob import glob
import math
from PIL import Image


def preprocess_image(path):
    image = np.array(Image.open(path).convert('RGB'))
    image = torch.tensor(image)
    image = image.permute(2, 0, 1) / 255.0
    return F.center_crop(image, (256, 256))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_images", type=str, required=True)
    parser.add_argument("--fake_images", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, rank=0, device_specific=False)

    device = torch.device('cuda:0')
    real_images = list(glob(os.path.join(args.real_images, '**', '*.*'), recursive=True))
    fake_images = list(glob(os.path.join(args.fake_images, '**', '*.*'), recursive=True))
    real_images = torch.stack([preprocess_image(i) for i in tqdm(real_images)])
    fake_images = torch.stack([preprocess_image(i) for i in tqdm(fake_images)])
    model = FrechetInceptionDistance(normalize=True)
    model = model.eval()
    model = model.to(device)
    model = model.to(torch.float32)
    for i in range(math.ceil(len(real_images)/args.batch_size)):
        model.update(real_images[i*args.batch_size: (i+1)*args.batch_size].to(device), real=True)
    for i in range(math.ceil(len(fake_images)/args.batch_size)):
        model.update(fake_images[i*args.batch_size: (i+1)*args.batch_size].to(device), real=False)


    scores = model.compute()
    print(f'Overall: {scores:4f}')

