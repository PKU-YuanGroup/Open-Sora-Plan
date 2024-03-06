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
from einops import rearrange
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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

from datasets import video_transforms, UCF101
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = torch.device('cuda:0')
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    # args.data_path = "E:/UCF-101"
    args.frame_interval = 3
    args.num_frames = 16

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(os.path.join(args.features_path, args.features_name), exist_ok=True)
    # os.makedirs(os.path.join(args.features_path, 'imagenet256_labels'), exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", cache_dir='cache_dir').to(device)

    # Setup data:

    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1
    transform_ucf101 = transforms.Compose([
        video_transforms.ToTensorVideo(),  # TCHW
        # video_transforms.RandomHorizontalFlipVideo(),
        video_transforms.UCFCenterCropVideo(args.image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)


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
        # x = x.to(device)
        x = x['video'].to(device, non_blocking=True)
        # y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            b, _, _, _, _ = x.shape
            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
            B = x.shape[0]
            n_sample = 64
            n_round = B // n_sample
            x_enc = []
            print(x.shape)
            for i in range(n_round+1):
                if i*n_sample == x.shape[0]:
                    break
                x_ = vae.encode(x[i*n_sample:(i+1)*n_sample]).latent_dist.sample().mul_(0.18215)
                x_enc.append(x_)
            x = torch.cat(x_enc, dim=0)
            x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()



        #     b, f, c, h, w = x.shape
        #     samples = rearrange(x, 'b f c h w -> (b f) c h w')
        #     samples_dec = []
        #     for i in tqdm(range(n_round+1)):
        #         samples_ = vae.decode(samples[i*n_sample:(i+1)*n_sample] / 0.18215).sample
        #         samples_dec.append(samples_)
        #     samples = torch.cat(samples_dec, dim=0)
        #
        #     samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
        #
        # video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        #
        # imageio.mimwrite('1.mp4', video_, fps=8, quality=9)
        # sys.exit()

        x = x.detach().cpu().numpy()  # (1, 4, 32, 32)
        np.save(f'{args.features_path}/{args.features_name}/{train_steps}.npy', x)

        # y = y.detach().cpu().numpy()  # (1,)
        # np.save(f'{args.features_path}/imagenet256_labels/{train_steps}.npy', y)

        train_steps += 1
        # print(train_steps)


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str)
    # parser.add_argument("--features-path", type=str, default="features")
    # parser.add_argument("--features-name", type=str, default="512video")
    # parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=512)
    # parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--epochs", type=int, default=1400)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    # parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    # parser.add_argument("--num-workers", type=int, default=0)
    # parser.add_argument("--log-every", type=int, default=100)
    # parser.add_argument("--ckpt-every", type=int, default=50_000)
    # args = parser.parse_args()
    # main(args)
    device = torch.device('cuda:0')
    x = torch.from_numpy(np.load('1.npy')).to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse", cache_dir='cache_dir').to(device)
    vae.eval()
    b, f, c, h, w = x.shape

    n_sample = 1
    n_round = b*f // n_sample
    samples = rearrange(x, 'b f c h w -> (b f) c h w')
    samples_dec = []

    for i in tqdm(range(n_round+1)):
        with torch.no_grad():
            samples_ = vae.decode(samples[i*n_sample:(i+1)*n_sample] / 0.18215).sample
        samples_dec.append(samples_.cpu())
    samples = torch.cat(samples_dec, dim=0)

    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)

    video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()

    imageio.mimwrite('1.mp4', video_, fps=10, quality=9)


