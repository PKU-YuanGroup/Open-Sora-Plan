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
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from opensora.dataset import ToTensorVideo, ae_norm, ae_denorm, CenterCropResizeVideo
from opensora.dataset.extract_feature_dataset import ExtractImage2Feature
from opensora.models.ae import getae

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import torch.distributed as dist




def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup a feature folder:
    if rank == 0:
        os.makedirs(f'{args.vis_path}/{args.features_name}', exist_ok=True)

    # Create model:
    ae = getae(args).to(device)
    # Setup data:
    norm_fun = ae_norm[args.ae]
    transform = Compose(
        [
            ToTensorVideo(),  # TCHW
            CenterCropResizeVideo(args.image_size),
            norm_fun,
        ]
    )
    dataset = ExtractImage2Feature(args, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    train_steps = 0
    for x, path in tqdm(loader):
        # x = x
        p = path[0].split('.')
        p = '.'.join(p[:-1]) + f'_{args.features_name}.npy'
        if os.path.exists(p):
            continue
        with torch.no_grad():
            b, _, _, _, _ = x.shape  # b t c h w
            assert b == 1
            x = x.to(device, non_blocking=True)
            x = x.transpose(1, 2).contiguous()  # B T C H W ->   # B C T H W
            x = ae.encode(x).cpu()  # b t c h w

            if train_steps == 0:
                b, t, c, h, w = x.shape
                samples = rearrange(x, 'b t c h w -> (b t) c h w')
                samples_dec = []
                samples_ = ae.decode(samples.to(device, non_blocking=True))
                samples_dec.append(samples_.cpu())
                samples = torch.cat(samples_dec, dim=0)

                samples = rearrange(samples, '(b t) c h w -> b t c h w', b=b)

                video_ = (ae_denorm[args.ae](samples[0]) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()

                imageio.mimwrite(f'{args.vis_path}/{args.features_name}/{train_steps}.mp4', video_, fps=30, quality=9)

        x = x.detach().cpu().numpy()[0][0]  # (1, 4, 32, 32)
        print(3333333, x.shape)
        sys.exit()
        np.save(p, x)

        # y = y.detach().cpu().numpy()  # (1,)
        # np.save(f'{args.features_path}/{args.features_name}/{train_steps}.npy', y)

        train_steps += 1
        # print(train_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--features-name", type=str, default="sky")
    parser.add_argument("--vis-path", type=str, default="extract_vis")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-sample", type=int, default=4)
    args = parser.parse_args()
    main(args)

