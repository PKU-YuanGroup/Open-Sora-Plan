# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""


import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from opensora.models.diffusion import get_t2v_models
from opensora.dataset import getdataset
from opensora.models.diffusion.diffusion import create_diffusion_T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from opensora.utils.utils import (create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard,   
                   write_tensorboard, setup_distributed,
                   get_experiment_dir)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.video_length) + 'S' + str(args.sample_rate)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        #OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    model = get_t2v_models(args)  #here load from pretrain_weight not checkpoint
    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    
    requires_grad(ema, False)
    diffusion = create_diffusion_T(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)


    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
    
        if "ema" in checkpoint:  # supports checkpoints from train.py
            print('Using Ema!')
            checkpoint = checkpoint["ema"]
        else:
            print('Using model!')
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))
    
    #use compile
    if args.use_compile:
        model = torch.compile(model)
        ema = torch.compile(ema)
    #mixed_precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])

    
    #load T5
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(device)

    #logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Setup data:
    dataset = getdataset(args)

    sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True,
    seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
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
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
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
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch
    """
    # in this origin version, we load the pretrained weight from latte
    #thus the checkpoint not the temp checkpoint for training
    #this loading is unnecessary
    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])
    """
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['videos'].to(device, non_blocking=True)
            
            # Map input images to latent space + normalize latents:
            b, _, _, _, _ = x.shape
            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            #x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
            x = rearrange(x, '(b f) c h w -> b c f h w', b=b).contiguous()   # match T2V model the input shape of the T2V is not equal to class condition model
            text = video_data['text']
            text_inputs = tokenizer(
                text, max_length=120, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True,
                add_special_tokens=True,
            )
            
            text_input_ids = text_inputs.input_ids.to(device)
            text_eme = text_encoder(text_input_ids, attention_mask=text_inputs.attention_mask.to(device))[0]
                
            model_kwargs = dict(encoder_hidden_states=text_eme)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                
            if scaler is None: #not mixed_precision
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_max_norm)
                opt.step()
            else: #mixed_precision
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.clip_max_norm)
                scaler.step(opt)
                scaler.update()

            lr_scheduler.step()
            opt.zero_grad()
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
                avg_loss = avg_loss.item() / dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save Latte checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        #"ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        # "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--max_train_steps", type=int, default=1000000)
    parser.add_argument("--local_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=2_000)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False)


    # --------------------------------------
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=3)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--save_ceph", action="store_true")
    parser.add_argument("--model", type=str, default='LatteT2V')
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--attention_bias", action="store_true")
    parser.add_argument("--fixed_spatial", action="store_true")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=16)

    parser.add_argument("--clip_max_norm", default=1, type=float, help="the maximum gradient norm (default None)")
    # --------------------------------------


    args = parser.parse_args()
    main(args)
