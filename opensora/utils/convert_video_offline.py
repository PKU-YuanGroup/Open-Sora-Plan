# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import sys

import imageio
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
import torchvision
from einops import rearrange
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

from opensora.dataset import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop, UCF101
from opensora.dataset.ucf101 import DecordInit
from opensora.models.ae import videovqvae, videovae, vqvae, getae, vae

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os



class FolderExtract(Dataset):
    def __init__(self, args, transform):
        self.data_path = args.data_path
        self.sample_rate = args.sample_rate
        self.transform = transform
        self.v_decoder = DecordInit()
        self.samples = list(glob(f'{self.data_path}/*mp4'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        video = self.decord_read(video_path)
        video = self.transform(video)  # T C H W -> T C H W
        return video

    def tv_read(self, path):
        vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit='sec', output_format='TCHW')
        total_frames = len(vframes)
        frame_indice = list(range(total_frames))[::self.sample_rate]
        video = vframes[frame_indice]
        return video

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        frame_indice = list(range(total_frames))[::self.sample_rate]
        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = torch.device('cuda:0')
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(os.path.join(args.features_path, args.features_name), exist_ok=True)

    # Create model:
    ae = getae(args).to(device)

    # Setup data:
    norm_fun = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    transform = Compose(
        [
            ToTensorVideo(),  # TCHW
            norm_fun,
        ]
    )
    dataset = FolderExtract(args, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x in tqdm(loader):
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            b, _, _, _, _ = x.shape  # b t c h w
            assert b == 1
            x = x[0]
            B = x.shape[0]
            n_sample = 32
            n_round = B // n_sample
            x_enc = []
            # print(x.shape)
            for i in range(n_round+1):
                if i*n_sample == x.shape[0]:
                    break
                x_ = ae.encode(x[i*n_sample:(i+1)*n_sample]) # B T C H W
                x_enc.append(x_)
            x = torch.cat(x_enc, dim=0)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=b).contiguous()


            if train_steps == 0:
                b, t, c, h, w = x.shape
                samples = rearrange(x, 'b t c h w -> (b t) c h w')
                samples_dec = []
                for i in range(n_round+1):
                    samples_ = ae.decode(samples[i*n_sample:(i+1)*n_sample])
                    samples_dec.append(samples_)
                samples = torch.cat(samples_dec, dim=0)

                samples = rearrange(samples, '(b t) c h w -> b t c h w', b=b)

                video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()

                imageio.mimwrite('test.mp4', video_, fps=8, quality=9)

        x = x.detach().cpu().numpy()  # (1, 4, 32, 32)
        np.save(f'{args.features_path}/{args.features_name}/{train_steps}.npy', x)

        # y = y.detach().cpu().numpy()  # (1,)
        # np.save(f'{args.features_path}/{args.features_name}/{train_steps}.npy', y)

        train_steps += 1
        # print(train_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--features-name", type=str, default="movie")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=3)
    args = parser.parse_args()
    main(args)

