# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional
import gc
import numpy as np
from einops import rearrange
from tqdm import tqdm
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
from torch.nn import functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer

import diffusers
from diffusers import DDPMScheduler, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available

from opensora.dataset import getdataset, ae_denorm
from opensora.models.ae import getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.diffusion import create_diffusion_T as create_diffusion
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.text_encoder import get_text_enc, get_text_warpper
from opensora.utils.dataset_utils import Collate
from opensora.models.ae import ae_stride_config, ae_channel_config
from opensora.models.diffusion import Diffusion_models
from opensora.sample.pipeline_videogen import VideoGenPipeline
from opensora.utils.utils import print_grad_norm

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)


@torch.inference_mode()
def log_validation(args, model, vae, text_encoder, tokenizer, accelerator, weight_dtype, global_step):
    validation_prompt = [
        "A quiet beach at dawn, the waves gently lapping at the shore and the sky painted in pastel hues.", 
        "The majestic beauty of a waterfall cascading down a cliff into a serene lake."
    ]
    logger.info(f"Running validation....\n")
    model = accelerator.unwrap_model(model)
    scheduler = PNDMScheduler()
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=model).to(device=accelerator.device)
    videos = []
    for prompt in validation_prompt:
        logger.info('Processing the ({}) prompt'.format(prompt))
        video = videogen_pipeline(prompt,
                                num_frames=args.num_frames,
                                height=args.max_image_size,
                                width=args.max_image_size,
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                enable_temporal_attentions=True,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                ).video
        videos.append(video[0])
    # import ipdb;ipdb.set_trace()
    gc.collect()
    torch.cuda.empty_cache()
    videos = torch.stack(videos).numpy()
    videos = rearrange(videos, 'b t h w c -> b t c h w')
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_videos = np.stack([np.asarray(vid) for vid in videos])
            tracker.writer.add_video("validation", np_videos, global_step, fps=24)
        if tracker.name == "wandb":
            import wandb
            tracker.log(
                {
                    "validation": [
                        wandb.Video(video, caption=f"{i}: {prompt}", fps=24)
                        for i, (video, prompt) in enumerate(zip(videos, validation_prompt))
                    ]
                }
            )

    del videogen_pipeline
    gc.collect()
    torch.cuda.empty_cache()
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Create model:
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    kwargs = {}
    ae = getae_wrapper(args.ae)(args.ae_path, cache_dir=args.cache_dir, **kwargs).eval()
    if args.enable_tiling:
        ae.vae.enable_tiling()
        ae.vae.tile_overlap_factor = args.tile_overlap_factor
        
    kwargs = {'load_in_8bit': args.enable_8bit_t5, 'torch_dtype': weight_dtype, 'low_cpu_mem_usage': True}
    text_enc = get_text_warpper(args.text_encoder_name)(args, **kwargs).eval()

    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    ae.vae_scale_factor = ae_stride_config[args.ae]
    assert ae_stride_h == ae_stride_w, f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert patch_size_h == patch_size_w, f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    # assert args.num_frames % ae_stride_t == 0, f"Num_frames must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert args.max_image_size % ae_stride_h == 0, f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}),  ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h
    latent_size = (args.max_image_size // ae_stride_h, args.max_image_size // ae_stride_w)
    ae.latent_size = latent_size

    if getae_wrapper(args.ae) == CausalVQVAEModelWrapper or getae_wrapper(args.ae) == CausalVAEModelWrapper:
        args.video_length = video_length = args.num_frames // ae_stride_t + 1
    else:
        video_length = args.num_frames // ae_stride_t
    model = Diffusion_models[args.model](
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae] * 2,
        # caption_channels=4096,
        # cross_attention_dim=1152,
        attention_bias=True,
        sample_size=latent_size,
        num_vector_embeds=None,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        use_linear_projection=False,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
        # norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        attention_type='default',
        video_length=video_length,
        attention_mode=args.attention_mode,
        compress_kv_factor=args.compress_kv_factor, 
        use_rope=args.use_rope, 
        model_max_length=args.model_max_length, 
    )
    model.gradient_checkpointing = args.gradient_checkpointing

    # # use pretrained model?
    if args.pretrained:
        if 'safetensors' in args.pretrained:
            from safetensors.torch import load_file as safe_load
            checkpoint = safe_load(args.pretrained, device="cpu")
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')['model']
        model_state_dict = model.state_dict()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        logger.info(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        logger.info(f'Successfully load {len(model.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')

    # Freeze vae and text encoders.
    ae.requires_grad_(False)
    text_enc.requires_grad_(False)
    # Set model as trainable.
    model.train()


    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    # ae.to(accelerator.device, dtype=torch.float32)
    ae.to(accelerator.device, dtype=weight_dtype)
    # ae.to(accelerator.device)
    text_enc.to(accelerator.device, dtype=weight_dtype)
    # text_enc.to(accelerator.device)

    # Create EMA for the unet.
    if args.use_ema:
        ema_model = deepcopy(model)
        ema_model = EMAModel(ema_model.parameters(), model_cls=LatteT2V, model_config=ema_model.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "model"))
                    if weights:  # Don't pop if empty
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), LatteT2V)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = LatteT2V.from_pretrained(input_dir, subfolder="model")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = model.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data:
    train_dataset = getdataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.output_dir, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, (x, attn_mask, input_ids, cond_mask) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Sample noise that we'll add to the latents
                if not args.multi_scale:
                    assert torch.all(attn_mask)
                
                x = x.to(accelerator.device, dtype=weight_dtype)  # B C T+num_images H W, 16 + 4
                attn_mask = attn_mask.to(accelerator.device)  # B 1+num_images L
                # assert torch.all(attn_mask != 0), 'attn_mask must all 1'
                input_ids = input_ids.to(accelerator.device)  # B 1+num_images L
                cond_mask = cond_mask.to(accelerator.device)  # B 1+num_images L
                # print('x.shape, attn_mask.shape, input_ids.shape, cond_mask.shape', x.shape, attn_mask.shape, input_ids.shape, cond_mask.shape)
                
                with torch.no_grad():
                    # use for loop to avoid OOM, because T5 is too huge...
                    B, _, _ = input_ids.shape  # B T+num_images L  b 1+4, L
                    cond = torch.stack([text_enc(input_ids[i], cond_mask[i]) for i in range(B)])  # B 1+num_images L D
                    
                    # Map input images to latent space + normalize latents
                    if args.use_image_num == 0:
                        x = ae.encode(x)  # B C T H W
                    else:
                        videos, images = x[:, :, :-args.use_image_num], x[:, :, -args.use_image_num:]
                        videos = ae.encode(videos)  # B C T H W


                        def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
                            from examples.rec_imvi_vae import array_to_video
                            x = x.detach().cpu()
                            x = torch.clamp(x, -1, 1)
                            x = (x + 1) / 2
                            x = x.permute(1, 2, 3, 0).numpy()
                            x = (255*x).astype(np.uint8)
                            array_to_video(x, fps=fps, output_file=output_file)
                            return

                        # videos = ae.decode(videos.to(dtype=weight_dtype))[0]
                        # videos = videos.transpose(0, 1)
                        # custom_to_video(videos.to(torch.float32), fps=24, output_file='tmp.mp4')
                        # sys.exit()

                        images = rearrange(images, 'b c t h w -> (b t) c 1 h w')
                        images = ae.encode(images)

                        # import ipdb;ipdb.set_trace()
                        # images = ae.decode(images.to(dtype=weight_dtype))
                        # for idx in range(args.use_image_num):
                        #     x = images[idx, 0, :, :, :].to(torch.float32)
                        #     x = x.squeeze()
                        #     x = x.detach().cpu().numpy()
                        #     x = np.clip(x, -1, 1)
                        #     x = (x + 1) / 2
                        #     x = (255 * x).astype(np.uint8)
                        #     x = x.transpose(1, 2, 0)
                        #     from PIL import Image
                        #     image = Image.fromarray(x)
                        #     image.save(f'tmp{idx}.jpg')
                        # import sys
                        # sys.exit()


                        images = rearrange(images, '(b t) c 1 h w -> b c t h w', t=args.use_image_num)
                        x = torch.cat([videos, images], dim=2)   #  b c 17+4, h, w

                

                # print('(x.shape, attn_mask.shape, cond.shape, cond_mask.shape', x.shape, attn_mask.shape, cond.shape, cond_mask.shape)
                model_kwargs = dict(encoder_hidden_states=cond, attention_mask=attn_mask,
                                    encoder_attention_mask=cond_mask, use_image_num=args.use_image_num)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=accelerator.device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)


                # accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                # print_grad_norm(model)
                # accelerator.deepspeed_engine_wrapped.engine.step()

                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if args.use_deepspeed or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())

                    if args.enable_tracker:
                        log_validation(args, model, ae, text_enc.text_enc, train_dataset.tokenizer, accelerator, weight_dtype, global_step)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--video_data", type=str, required='')
    parser.add_argument("--image_data", type=str, default='')
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=300)

    parser.add_argument('--enable_8bit_t5', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=['xformers', 'math', 'flash'], default="xformers")
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--compress_kv_factor', type=int, default=1)

    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="Latte-XL/122")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')

    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.5)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")

    args = parser.parse_args()
    main(args)
