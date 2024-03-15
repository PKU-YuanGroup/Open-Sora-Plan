# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import math

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
from accelerate import Accelerator
from torch import nn

from opensora.utils.utils import get_experiment_dir, create_logger, requires_grad, update_ema, write_tensorboard, \
    cleanup, create_tensorboard

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from opensora.dataset import getdataset
from opensora.models.ae import getae
from opensora.utils.dataset_utils import Collate
from opensora.models.ae import ae_stride_config, ae_channel_config
from opensora.models.diffusion import Diffusion_models
from opensora.models.diffusion.diffusion import create_diffusion

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    # Setup an experiment folder:
    if args.resume_from_checkpoint:
        experiment_dir = f"{args.results_dir}/{args.resume_from_checkpoint}"
    else:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/122 --> Latte-XL-122 (for naming folders)
        ae_string_name = args.ae.split("/")[-1]  # e.g., stabilityai/sd-vae-ft-mse --> sd-vae-ft-mse (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.sample_rate)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{ae_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f'{args}')
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert ae_stride_h == ae_stride_w, f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    assert patch_size_h == patch_size_w, f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    assert args.num_frames % ae_stride_t == 0, f"Num_frames must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert args.max_image_size % ae_stride_h == 0, f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}),  ae_stride_h ({ae_stride_h})."

    latent_size = (args.max_image_size // ae_stride_h, args.max_image_size // ae_stride_w)

    model = Diffusion_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=ae_channel_config[args.ae],
        extras=args.extras,
        num_frames=args.num_frames // ae_stride_t,
        attention_mode=args.attention_mode
    )
    model.gradient_checkpointing = args.gradient_checkpointing

    # # use pretrained model?
    if args.pretrained:
        # load from pixart-alpha
        pixelart_alpha = torch.load(args.pretrained, map_location='cpu')['state_dict']
        checkpoint = {}
        for k, v in pixelart_alpha.items():
            if 'x_embedder' in k or 't_embedder' in k:
                checkpoint[k] = v
            if k.startswith('blocks'):
                if 'cross_attn' not in k and 'drop_path' not in k and 'scale_shift_table' not in k:
                    if 'q_norm' not in k and 'k_norm' not in k:
                        blk_id = k[7]
                        new_k = f'blocks.{str(int(blk_id) * 2)}{k[8:]}'
                        checkpoint[new_k] = v
        total_load = set(model.state_dict().keys()) & set(checkpoint.keys())
        msg = model.load_state_dict(checkpoint, strict=False)
        logger.info(f'Successfully load {len(total_load)} keys from {args.pretrained}!')

    model = model.to(device)
    # Note that parameter initialization is done within the DiT constructor
    if args.use_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    if args.use_compile:
        model = torch.compile(model)

    ae = getae(args).to(device)
    if accelerator.is_main_process:
        logger.info(f"{model}")
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"Training Parameters: {n}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    dataset = getdataset(args)
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=Collate(args)
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    if args.use_ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    model, opt, loader, lr_scheduler = accelerator.prepare(model, opt, loader, lr_scheduler)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        state_dict = torch.load(os.path.join(os.path.join(experiment_dir, 'checkpoints'), path), map_location='cpu')
        print(state_dict['args'])
        state_dict = {'module.'+k: v for k, v in state_dict['model'].items()}
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state_dict(state_dict)
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch

    if args.pretrained:
        try:
            train_steps = int(args.pretrained.split("/")[-1].split('.')[0])
        except:
            train_steps = 0

    if accelerator.is_main_process:
        logger.info(f"Training for {num_train_epochs} epochs...")

    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    for epoch in range(first_epoch, num_train_epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for step, (x, y, attn_mask) in enumerate(loader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            x = x.to(device)  # B C T H W
            y = y.to(device)
            attn_mask = attn_mask.to(device)  # B T H W
            with torch.no_grad():
                # Map input images to latent space + normalize latents
                x = ae.encode(x)  # B C T H W -> B T C H W

            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                model_kwargs = dict(y=y, attn_mask=attn_mask)
            else:
                model_kwargs = dict(y=None, attn_mask=attn_mask)
            with torch.autocast(device_type='cuda', dtype=dtype):
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                opt.zero_grad()

            accelerator.backward(loss)

            # if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
            #     gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
            # else:
            #     gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

            opt.step()

            if args.use_ema:
                update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / accelerator.num_processes
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if accelerator.is_main_process:
                    # logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    logger.info(
                        f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save Model checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    if args.use_ema:
                        checkpoint.update({"ema": ema.state_dict()})
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="DiT-XL/122")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--max-train-steps", type=int, default=1000000)
    parser.add_argument("--local-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)


    # --------------------------------------
    parser.add_argument("--ae", type=str, default="ucf101_stride4x4x4")
    parser.add_argument("--extras", type=int, default=2, choices=[1, 2, 78])
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-image-size", type=int, default=128)
    parser.add_argument("--dynamic-frames", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--attention-mode", type=str, choices=['xformers', 'math', 'flash'], default="math")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=[None, "fp16", "bf16"])
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    # --------------------------------------


    args = parser.parse_args()
    main(args)
