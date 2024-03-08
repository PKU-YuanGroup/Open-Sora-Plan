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
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from opensora.dataset import ToTensorVideo, ae_norm, ae_denorm, CenterCropResizeVideo
from opensora.dataset.extract_feature_dataset import ExtractVideo2Feature
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
    dataset = ExtractVideo2Feature(args, transform=transform)

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
        with torch.no_grad():
            b, t, _, _, _ = x.shape  # b t c h w
            assert b == 1
            n_frame = args.n_frame_per_sample * args.sample_rate
            n_round = t // n_frame
            x_enc = []  # n B T C H W
            for i in range(n_round+1):
                if i*n_frame == x.shape[1] or x.shape[1] - i*n_frame < n_frame:
                    break
                x_ = x[:, i*n_frame:(i+1)*n_frame][:, ::args.sample_rate].to(device, non_blocking=True)
                x_ = x_.permute(0, 2, 1, 3, 4).contiguous()  # B T C H W -> B C T H W
                x_ = ae.encode(x_)  # B C T H W -> B T C H W
                x_enc.append(x_.cpu())

            if train_steps == 0:
                for idx, samples in enumerate(x_enc):
                    samples = ae.decode(samples.to(device, non_blocking=True)).cpu()  # b t c h w
                    video_ = (ae_denorm[args.ae](samples[0]) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                    imageio.mimwrite(f'{args.vis_path}/{args.features_name}/{train_steps}-{idx}.mp4', video_, fps=30//args.sample_rate, quality=9)

        for idx, x in enumerate(x_enc):
            x = x.detach().numpy()[0]  # (t, 4, 32, 32)
            p = path[0].split('.')
            p = '.'.join(p[:-1]) + f'_{args.features_name}_{idx}.npy'
            if not os.path.exists(p):
                np.save(p, x)

        train_steps += 1
        # print(train_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ae", type=str, default="bair_stride4x2x2")
    parser.add_argument("--features-name", type=str, default="landscope")
    parser.add_argument("--vis-path", type=str, default="extract_vis")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=3)
    parser.add_argument("--n-frame-per-sample", type=int, default=128)
    args = parser.parse_args()
    main(args)

