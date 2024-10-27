# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import deepspeed
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
import torch.utils
import torch.utils.data
from tqdm import tqdm
import time

from opensora.adaptor.modules import replace_with_fp32_forwards
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import initialize_sequence_parallel_state, \
        destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state
    from opensora.acceleration.communications import prepare_parallel_data, broadcast
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import initialize_sequence_parallel_state, \
        destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state
    from opensora.utils.communications import prepare_parallel_data, broadcast
    pass

from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
from torch.nn import functional as F
import transformers
from transformers.utils import ContextManagers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed, DeepSpeedPlugin
from accelerate.state import AcceleratorState
from packaging import version
from tqdm.auto import tqdm

import copy
import diffusers
from diffusers import DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.models import CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_warpper
from opensora.dataset import getdataset
from opensora.models import CausalVAEModelWrapper
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
from opensora.utils.dataset_utils import Collate, LengthGroupedSampler
from opensora.utils.ema_utils import EMAModel
from opensora.utils.utils import explicit_uniform_sampling, get_common_weights
from opensora.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper

# from opensora.utils.utils import monitor_npu_power

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)
GB = 1024 * 1024 * 1024

@torch.inference_mode()
def log_validation(args, model, vae, text_encoder, tokenizer, accelerator, weight_dtype, global_step, ema=False):
    positive_prompt = "(masterpiece), (best quality), (ultra-detailed), {}. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
    negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, 
                        """
    validation_prompt = [
        "a cat wearing sunglasses and working as a lifeguard at pool.",
        "A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell, is the main focus of the video, swimming gracefully towards the right side of the frame. The coral reef, teeming with life, is visible in the background, providing a vibrant and colorful backdrop to the turtle's journey. Several small fish, darting around the turtle, add a sense of movement and dynamism to the scene."
        ]
    logger.info(f"Running validation....\n")
    model = accelerator.unwrap_model(model)
    scheduler = DPMSolverMultistepScheduler()
    opensora_pipeline = OpenSoraPipeline(vae=vae,
                                         text_encoder_1=text_encoder[0],
                                         text_encoder_2=text_encoder[1],
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=model).to(device=accelerator.device)
    videos = []
    for prompt in validation_prompt:
        logger.info('Processing the ({}) prompt'.format(prompt))
        video = opensora_pipeline(
                                positive_prompt.format(prompt),
                                negative_prompt=negative_prompt, 
                                num_frames=args.num_frames,
                                height=args.max_height,
                                width=args.max_width,
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=args.guidance_scale,
                                enable_temporal_attentions=True,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                max_sequence_length=args.model_max_length,
                                ).images
        videos.append(video[0])
    # import ipdb;ipdb.set_trace()
    gc.collect()
    torch.cuda.empty_cache()
    videos = torch.stack(videos).numpy()
    videos = rearrange(videos, 'b t h w c -> b t c h w')
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            if videos.shape[1] == 1:
                assert args.num_frames == 1
                images = rearrange(videos, 'b 1 c h w -> (b 1) h w c')
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(f"{'ema_' if ema else ''}validation", np_images, global_step, dataformats="NHWC")
            else:
                np_videos = np.stack([np.asarray(vid) for vid in videos])
                tracker.writer.add_video(f"{'ema_' if ema else ''}validation", np_videos, global_step, fps=24)
        if tracker.name == "wandb":
            import wandb
            if videos.shape[1] == 1:
                images = rearrange(videos, 'b 1 c h w -> (b 1) h w c')
                logs = {
                    f"{'ema_' if ema else ''}validation": [
                        wandb.Image(image, caption=f"{i}: {prompt}")
                        for i, (image, prompt) in enumerate(zip(images, validation_prompt))
                    ]
                }
            else:
                logs = {
                    f"{'ema_' if ema else ''}validation": [
                        wandb.Video(video, caption=f"{i}: {prompt}", fps=24)
                        for i, (video, prompt) in enumerate(zip(videos, validation_prompt))
                    ]
                }
            tracker.log(logs, step=global_step)

    del opensora_pipeline
    gc.collect()
    torch.cuda.empty_cache()

class ProgressInfo:
    def __init__(
        self, global_step, train_loss=0.0, max_grad_norm=0.0, weight_norm=0.0, 
        moving_avg_max_grad_norm=-1e6, moving_avg_max_grad_norm_var=3.0, 
        clip_coef=1.0, max_grad_norm_clip=0.0, max_norm=1.0, max_grad_norm_var=0.0, 
        detect_nan=0.0, max_timesteps=1000.0, min_timesteps=1.0, max_train_loss=0.0, 
        ):
        self.global_step = global_step
        self.train_loss = train_loss
        self.max_grad_norm = max_grad_norm
        self.weight_norm = weight_norm
        self.moving_avg_max_grad_norm = moving_avg_max_grad_norm
        self.moving_avg_max_grad_norm_var = moving_avg_max_grad_norm_var
        self.clip_coef = clip_coef
        self.max_grad_norm_clip = max_grad_norm_clip
        self.max_norm = max_norm
        self.max_grad_norm_var = max_grad_norm_var
        self.detect_nan = detect_nan
        self.max_timesteps = max_timesteps
        self.min_timesteps = min_timesteps
        self.max_train_loss = max_train_loss


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch_npu is not None and npu_config is not None:
        npu_config.print_msg(args)
        npu_config.seed_everything(args.seed)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    deepspeed_plugins = {}
    # https://huggingface.co/docs/accelerate/v1.0.1/en/usage_guides/deepspeed_multiple_model
    train_plugin = DeepSpeedPlugin(hf_ds_config=args.train_deepspeed_config_file)
    deepspeed_plugins["train"] = train_plugin
    
    if args.use_ema:
        # Use FP32 for EMA model
        # DeepSpeed will raise an error if train_micro_batch_size_per_gpu isn't specified, even if this particular model isn't being trained.
        ema_plugin = DeepSpeedPlugin(hf_ds_config=args.eval_deepspeed_config_file)
        ema_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
        ema_plugin.deepspeed_config["bf16"]["enabled"] = args.ema_bf16
        deepspeed_plugins["ema"] = ema_plugin

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugins=deepspeed_plugins, 
    )

    if args.skip_abnorml_step and accelerator.distributed_type == DistributedType.DEEPSPEED:
        from opensora.utils.deepspeed_utils import backward
        deepspeed.runtime.engine.DeepSpeedEngine.backward = backward

    if args.num_frames != 1:
        initialize_sequence_parallel_state(args.sp_size)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        wandb_init_kwargs = {"wandb": {"name": args.log_name or args.proj_name or args.output_dir}}

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.basename(args.proj_name or args.output_dir), config=vars(args), 
                                  init_kwargs=wandb_init_kwargs if args.report_to == "wandb" else None)

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
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    torch.distributed.barrier()

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    checkpoint_path, global_step = None, 0
    initial_global_step_for_sampler = args.trained_data_global_step
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = args.resume_from_checkpoint
            if initial_global_step_for_sampler is None:
                initial_global_step_for_sampler = 0
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None
            if initial_global_step_for_sampler is None:
                initial_global_step_for_sampler = int(dirs[-1].split("-")[1]) if len(dirs) > 0 else 0

        if checkpoint_path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            global_step = int(checkpoint_path.split("-")[-1])

    if initial_global_step_for_sampler is None:
        initial_global_step_for_sampler = 0


    # =======================================================================================================
    # STEP 1: Create VAE model
    kwargs = {}
    ae = ae_wrapper[args.ae](args.ae_path, cache_dir=args.cache_dir, **kwargs).eval()
    if args.enable_tiling:
        ae.vae.enable_tiling()
    ae.vae.requires_grad_(False)
    if not args.post_to_device:
        ae.vae.to(accelerator.device, dtype=torch.float32 if args.vae_fp32 else weight_dtype)
        logger.info(f"Load VAE model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    # =======================================================================================================


    # =======================================================================================================
    # STEP 2: Create text encoder model
    text_enc_1 = get_text_warpper(args.text_encoder_name_1)(args).eval()
    text_enc_1.requires_grad_(False)
    if not args.post_to_device:
        text_enc_1.to(accelerator.device, dtype=weight_dtype)
        logger.info(f"Load text encoder model 1 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)

    text_enc_2 = None
    if args.text_encoder_name_2 is not None:
        text_enc_2 = get_text_warpper(args.text_encoder_name_2)(args).eval()
        text_enc_2.requires_grad_(False)
        if not args.post_to_device:
            text_enc_2.to(accelerator.device, dtype=weight_dtype)
            logger.info(f"Load text encoder model 2 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    # =======================================================================================================


    # =======================================================================================================
    # STEP 3: Prepare diffusion model (Trained model must be prepared first!)
    accelerator.state.select_deepspeed_plugin("train")

    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    ae.vae_scale_factor = (ae_stride_t, ae_stride_h, ae_stride_w)
    assert ae_stride_h == ae_stride_w, f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert patch_size_h == patch_size_w, f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    assert (args.num_frames - 1) % ae_stride_t == 0, f"(Frames - 1) must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert args.max_height % ae_stride_h == 0, f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
    assert args.max_width % ae_stride_h == 0, f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h
    ae.latent_size = latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    args.latent_size_t = latent_size_t = (args.num_frames - 1) // ae_stride_t + 1
    if checkpoint_path:
        model = Diffusion_models_class[args.model].from_pretrained(os.path.join(checkpoint_path, "model"))
        logger.info(f'Successully resume model from {os.path.join(checkpoint_path, "model")}', main_process_only=True)
    else:
        model = Diffusion_models[args.model](
            in_channels=ae_channel_config[args.ae],
            out_channels=ae_channel_config[args.ae],
            sample_size_h=latent_size,
            sample_size_w=latent_size,
            sample_size_t=latent_size_t,
            interpolation_scale_h=args.interpolation_scale_h,
            interpolation_scale_w=args.interpolation_scale_w,
            interpolation_scale_t=args.interpolation_scale_t,
            sparse1d=args.sparse1d, 
            sparse_n=args.sparse_n, 
            skip_connection=args.skip_connection, 
        )

    # use pretrained model?
    if checkpoint_path is None and args.pretrained:
        model_state_dict = model.state_dict()
        logger.info(f'Load from {args.pretrained}')
        if args.pretrained.endswith('.safetensors'):  
            from safetensors.torch import load_file as safe_load
            pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            checkpoint = get_common_weights(pretrained_checkpoint, model_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        elif os.path.isdir(args.pretrained):
            model_pretrained = Diffusion_models_class[args.model].from_pretrained(args.pretrained)
            pretrained_checkpoint = model_pretrained.state_dict()
            # checkpoint = get_common_weights(pretrained_checkpoint, model_state_dict)
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            model_pretrained = None
            del model_pretrained
        else:
            pretrained_checkpoint = torch.load(args.pretrained, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']
            checkpoint = get_common_weights(pretrained_checkpoint, model_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        logger.info(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        logger.info(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')
        del model_state_dict, checkpoint, missing_keys, unexpected_keys
        gc.collect()

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    logger.info(f"Load diffusion model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    # =======================================================================================================


    # =======================================================================================================
    # STEP 4: Create EMAModel
    if args.use_ema:
        ema_model_path = os.path.join(checkpoint_path, "model_ema")
        if checkpoint_path and os.path.exists(ema_model_path):
            ema_model = EMAModel.from_pretrained(ema_model_path, model_cls=Diffusion_models_class[args.model])
            logger.info(f'Successully resume EMAModel from {ema_model_path}', main_process_only=True)
        else:
            ema_model = EMAModel(deepcopy(model), decay=args.ema_decay, update_after_step=args.ema_start_step,
                                model_cls=Diffusion_models_class[args.model], model_config=model.config)
            logger.info(f'Successully deepcopy EMAModel from model', main_process_only=True)
        ema_model.model = ema_model.model.eval()
        ema_model.model.requires_grad_(False)
        ema_model.model.config.hidden_size = model.config.attention_head_dim * model.config.num_attention_heads

        logger.info(f"Load ema model 2 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    # =======================================================================================================


    # =======================================================================================================
    # STEP 5: Create Scheduler
    kwargs = dict(
        prediction_type=args.prediction_type, 
        rescale_betas_zero_snr=args.rescale_betas_zero_snr
    )
    if args.cogvideox_scheduler:
        noise_scheduler = CogVideoXDDIMScheduler(**kwargs)
    elif args.v1_5_scheduler:
        kwargs['beta_start'] = 0.00085
        kwargs['beta_end'] = 0.0120
        kwargs['beta_schedule'] = "scaled_linear"
        noise_scheduler = DDPMScheduler(**kwargs)
    elif args.rf_scheduler:
        noise_scheduler = FlowMatchEulerDiscreteScheduler()
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    else:
        noise_scheduler = DDPMScheduler(**kwargs)
    # =======================================================================================================

    # Enable TF32 for faster training on Ampere GPUs
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # =======================================================================================================
    # STEP 6: Create Optimizer

    params_to_optimize = model.parameters()
    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
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

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    logger.info(f"optimizer: {optimizer}")
    # =======================================================================================================
    

    # =======================================================================================================
    # # STEP 7: Create Dataset & Dataloader & LR_Scheduler

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
    args.total_batch_size = total_batch_size
    if args.max_hxw is not None and args.min_hxw is None:
        args.min_hxw = args.max_hxw // 4
    train_dataset = getdataset(args)
    sampler = LengthGroupedSampler(
                args.train_batch_size,
                world_size=accelerator.num_processes, 
                gradient_accumulation_size=args.gradient_accumulation_steps, 
                initial_global_step=initial_global_step_for_sampler, 
                lengths=train_dataset.lengths, 
                group_data=args.group_data, 
            )
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        pin_memory=False,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler, 
        drop_last=True, 
        # prefetch_factor=4
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
    # =======================================================================================================


    # =======================================================================================================
    # STEP 8: Prepare everything with our `accelerator`.
    
    # STEP 8.1: Prepare everything for training
    logger.info(f"Before accelerator.prepare, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
    model_config = model.config
    model_config_to_save = model.to_json_string()
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if checkpoint_path:
        accelerator.load_state(checkpoint_path)
    logger.info(f"After accelerator.prepare, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)

    if args.use_ema:
        # STEP 8.2: Prepare EMA model
        accelerator.state.select_deepspeed_plugin("ema")
        ema_model.model = accelerator.prepare(ema_model.model)
        ema_model.model.eval()
        logger.info(f"Prepare ema model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
        # STEP 8.3: All models have been prepared, start training
        accelerator.state.select_deepspeed_plugin("train")
    model.train()

    # FIXME: EMAModel from diffusers have bug, which can NOT resume ema_decay
    if args.use_ema:
        ema_model.decay = args.ema_decay
    # =======================================================================================================
    
    if args.post_to_device:
        ae.vae.to(accelerator.device, dtype=torch.float32 if args.vae_fp32 else weight_dtype)
        logger.info(f"Load VAE model finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
        text_enc_1.to(accelerator.device, dtype=weight_dtype)
        logger.info(f"Load text encoder model 1 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)
        if args.text_encoder_name_2 is not None:
            text_enc_2.to(accelerator.device, dtype=weight_dtype)
            logger.info(f"Load text encoder model 2 finish, memory_allocated: {torch.cuda.memory_allocated()/GB:.2f} GB", main_process_only=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # =======================================================================================================
    # Step 9: Train!
    logger.info("***** Running training *****")
    logger.info(f"  Model = {model}")
    logger.info(f'  Model config = {model_config}')
    logger.info(f"  Args = {args}")
    logger.info(f"  Noise_scheduler = {noise_scheduler}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total optimization steps (num_update_steps_per_epoch) = {num_update_steps_per_epoch}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
    
    logger.info(f"  AutoEncoder = {args.ae}; Dtype = {ae.vae.dtype}; Parameters = {sum(p.numel() for p in ae.parameters()) / 1e9} B")
    logger.info(f"  Text_enc_1 = {args.text_encoder_name_1}; Dtype = {weight_dtype}; Parameters = {sum(p.numel() for p in text_enc_1.parameters()) / 1e9} B")
    if args.text_encoder_name_2 is not None:
        logger.info(f"  Text_enc_2 = {args.text_encoder_name_2}; Dtype = {weight_dtype}; Parameters = {sum(p.numel() for p in text_enc_2.parameters()) / 1e9} B")
    if args.use_ema:
        logger.info(f"  EMA model = {type(ema_model.model)}; Dtype = {ema_model.model.dtype}; Parameters = {sum(p.numel() for p in ema_model.model.parameters()) / 1e9} B")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_info = ProgressInfo(global_step, train_loss=0.0, max_grad_norm=0.0)

    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)  # 0.001, 1
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)  # 1, 1000
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def sync_gradients_info(loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if args.use_ema and progress_info.global_step % args.ema_update_freq == 0:
            logger.info(f'Update ema with decay {ema_model.decay}, ema_update_freq {args.ema_update_freq}', main_process_only=True)
            ema_model.step(model.parameters())
            logger.info(f'Finish ema', main_process_only=True)
        progress_bar.update(1)
        progress_info.global_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        
        train_loss = progress_info.train_loss
        accelerator.log(
                {
                    "train_loss": train_loss, "max_grad_norm": progress_info.max_grad_norm, 
                    "weight_norm": progress_info.weight_norm, 
                    "max_grad_norm_clip": progress_info.max_grad_norm_clip, 
                    "moving_avg_max_grad_norm": progress_info.moving_avg_max_grad_norm, 
                    "moving_avg_max_grad_norm_var": progress_info.moving_avg_max_grad_norm_var, 
                    "max_norm": progress_info.max_norm, 
                    "num_zero_grad": progress_info.num_zero_grad, 
                    "detect_nan": progress_info.detect_nan, 
                    "clip_coef": progress_info.clip_coef,
                    "min_timesteps": progress_info.min_timesteps, 
                    "max_timesteps": progress_info.max_timesteps, 
                    "max_grad_norm_var": progress_info.max_grad_norm_var, 
                    "max_train_loss": progress_info.max_train_loss, 
                    "lr": lr_scheduler.get_last_lr()[0]
                }, step=progress_info.global_step
            )

        if torch_npu is not None and npu_config is not None:
            npu_config.print_msg(f"Step: [{progress_info.global_step}], local_loss={loss.detach().item()}, "
                                f"train_loss={train_loss}, max_grad_norm={progress_info.max_grad_norm}, max_grad_norm_clip={progress_info.max_grad_norm_clip}, "
                                f"weight_norm={progress_info.weight_norm}, time_cost={one_step_duration}",
                                rank=0)
        progress_info.train_loss = 0.0
        progress_info.max_grad_norm = 0.0
        progress_info.weight_norm = 0.0
        progress_info.num_zero_grad = 1.0
        progress_info.max_grad_norm_clip = 0.0
        progress_info.clip_coef = 1.0
        progress_info.max_norm = 1.0
        progress_info.max_grad_norm_var = 0.0
        progress_info.detect_nan = 0.0
        progress_info.max_train_loss = 0.0
        progress_info.min_timesteps = 1.0
        progress_info.max_timesteps = 1000.0

        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
            if progress_info.global_step % args.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if accelerator.is_main_process and args.checkpoints_total_limit is not None:
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

                save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}")
                # FIXME: https://github.com/huggingface/accelerate/issues/3140
                acc_models = accelerator._models
                accelerator._models = [model for model in acc_models if model.checkpoint_engine is not None]
                accelerator.save_state(save_path)
                accelerator._models = acc_models
                
                model_path = os.path.join(save_path, "model")
                os.makedirs(model_path, exist_ok=True)
                with open(os.path.join(model_path, 'config.json'), "w", encoding="utf-8") as writer:
                    writer.write(model_config_to_save)
                convert_zero_checkpoint_to_fp32_state_dict(save_path, os.path.join(model_path, 'diffusion_pytorch_model.safetensors'))

                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(save_path, "model_ema"))
                logger.info(f"Saved state to {save_path}")

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

    def run(step_, model_input, model_kwargs, prof):
        # print("rank {} | step {} | cd run fun".format(accelerator.process_index, step_))
        global start_time
        start_time = time.time()

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        if not args.rf_scheduler:
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn((model_input.shape[0], model_input.shape[1], 1, 1, 1),
                                                        device=model_input.device)

            # Sample a random timestep for each image without bias.
            timesteps = explicit_uniform_sampling(
                T=noise_scheduler.config.num_train_timesteps, 
                n=accelerator.num_processes, 
                rank=accelerator.process_index, 
                bsz=bsz, device=model_input.device, 
                )
            if get_sequence_parallel_state():  # image do not need sp, disable when image batch
                broadcast(timesteps)

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        model_pred = model(
            noisy_model_input,
            timesteps,
            **model_kwargs
        )[0]
        mask = model_kwargs.get('attention_mask', None)
        if not args.rf_scheduler:
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            elif noise_scheduler.config.prediction_type == "sample":
                # We set the target to latents here, but the model_pred will return the noise sample prediction.
                target = model_input
                # We will have to subtract the noise residual from the prediction to get the target sample.
                model_pred = model_pred - noise
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if get_sequence_parallel_state():
                if torch.all(mask.bool()):
                    mask = None
                # mask    (sp_bs*b t h w)
                assert mask is None
            b, c, _, _, _ = model_pred.shape
            if mask is not None:
                mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()  # b t h w -> b c t h w
                mask = mask.reshape(b, -1)
            if args.snr_gamma is None:
                # model_pred: b c t h w, attention_mask: b t h w
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.reshape(b, -1)
                if mask is not None:
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on unpad patches
                else:
                    loss = loss.mean()
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                    dim=1
                )[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                else:
                    raise NameError(f'{noise_scheduler.config.prediction_type}')
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.reshape(b, -1)
                mse_loss_weights = mse_loss_weights.reshape(b, 1)
                if mask is not None:
                    loss = (loss * mask * mse_loss_weights).sum() / mask.sum()  # mean loss on unpad patches
                else:
                    loss = (loss * mse_loss_weights).mean()
        else:
            if torch.all(mask.bool()):
                mask = None

            b, c, _, _, _ = model_pred.shape
            if mask is not None:
                mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()  # b t h w -> b c t h w
                mask = mask.reshape(b, -1)

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            target = noise - model_input

            # Compute regular loss.
            loss_mse = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
            if mask is not None:
                loss = (loss_mse * mask).sum() / mask.sum()
            else:
                loss = loss_mse.mean()
        # Gather the losses across all processes for logging (if we use distributed training).
        # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        # progress_info.train_loss += loss.detach().item() / args.gradient_accumulation_steps
        timesteps_list = accelerator.gather(timesteps)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f'Detect loss error, timestep {timesteps_list}')
        max_timesteps = timesteps_list.max().item()
        min_timesteps = timesteps_list.min().item()
        # Backpropagate
        if args.skip_abnorml_step and accelerator.distributed_type == DistributedType.DEEPSPEED:
            results = accelerator.deepspeed_engine_wrapped.engine.backward(
                loss, 
                moving_avg_max_grad_norm=progress_info.moving_avg_max_grad_norm, 
                ema_decay_grad_clipping=args.ema_decay_grad_clipping, 
                moving_avg_max_grad_norm_var=progress_info.moving_avg_max_grad_norm_var, accelerator=accelerator,
                force_zero_grad_step=args.force_zero_grad_step, step_=step_ 
                )
            _, max_grad_norm, weight_norm, moving_avg_max_grad_norm, max_grad_norm_clip, max_norm, \
                moving_avg_max_grad_norm_var, max_grad_norm_var, num_zero_grad, detect_nan, clip_coef, zero_grad_list = results
                
            # print('rank {} | step {} | max_grad_norm {}'.format(accelerator.process_index, step_, max_grad_norm))
            progress_info.max_grad_norm += max_grad_norm / args.gradient_accumulation_steps
            progress_info.weight_norm += weight_norm / args.gradient_accumulation_steps
            progress_info.moving_avg_max_grad_norm = moving_avg_max_grad_norm / args.gradient_accumulation_steps
            progress_info.moving_avg_max_grad_norm_var = moving_avg_max_grad_norm_var / args.gradient_accumulation_steps
            progress_info.num_zero_grad = num_zero_grad
            progress_info.max_norm = max_norm
            progress_info.clip_coef = clip_coef
            progress_info.detect_nan = detect_nan
            progress_info.max_timesteps = max_timesteps
            progress_info.min_timesteps = min_timesteps
            progress_info.max_grad_norm_clip = max_grad_norm_clip / args.gradient_accumulation_steps
            progress_info.max_grad_norm_var += max_grad_norm_var / args.gradient_accumulation_steps
            
            accelerator.deepspeed_engine_wrapped.engine.step()

            avg_loss_list = accelerator.gather(loss)
            progress_info.max_train_loss = avg_loss_list.max().detach().item()
            avg_loss_list = (1.0 - zero_grad_list.float()) * avg_loss_list
            avg_loss_list = avg_loss_list / clip_coef
        else:
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = model.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            avg_loss_list = accelerator.gather(loss)

        progress_info.train_loss += avg_loss_list.mean().detach().item() / args.gradient_accumulation_steps
        optimizer.zero_grad()
        lr_scheduler.step()
        if accelerator.sync_gradients:
            sync_gradients_info(loss)

        if accelerator.is_main_process:
            if progress_info.global_step % args.checkpointing_steps == 0:
                if args.enable_tracker:
                    log_validation(
                        args, model, ae, [text_enc_1.text_enc, getattr(text_enc_2, 'text_enc', None)], 
                        train_dataset.tokenizer, accelerator, weight_dtype, progress_info.global_step
                    )

                    if args.use_ema and npu_config is None:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                        log_validation(
                            args, model, ae, [text_enc_1.text_enc, getattr(text_enc_2, 'text_enc', None)], 
                            train_dataset.tokenizer, accelerator, weight_dtype, progress_info.global_step, ema=True
                        )
                        # Switch back to the original UNet parameters.
                        ema_model.restore(model.parameters())

        if prof is not None:
            prof.step()

        return loss

    def train_one_step(step_, data_item_, prof_=None):
        train_loss = 0.0
        x, attn_mask, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2 = data_item_
        # print(f'step: {step_}, rank: {accelerator.process_index}, x: {x.shape}, dtype: {x.dtype}')

        x = x.to(accelerator.device, dtype=ae.vae.dtype, non_blocking=True)  # B C T H W
        attn_mask = attn_mask.to(accelerator.device, non_blocking=True)  # B T H W
        input_ids_1 = input_ids_1.to(accelerator.device, non_blocking=True)  # B 1 L
        cond_mask_1 = cond_mask_1.to(accelerator.device, non_blocking=True)  # B 1 L
        input_ids_2 = input_ids_2.to(accelerator.device, non_blocking=True) if input_ids_2 is not None else input_ids_2 # B 1 L
        cond_mask_2 = cond_mask_2.to(accelerator.device, non_blocking=True) if cond_mask_2 is not None else cond_mask_2 # B 1 L
        
        with torch.no_grad():
            B, N, L = input_ids_1.shape  # B 1 L
            # use batch inference
            input_ids_1 = input_ids_1.reshape(-1, L)
            cond_mask_1 = cond_mask_1.reshape(-1, L)
            if args.random_data:
                cond_1 = torch.rand(B, L, 2048, device=x.device, dtype=weight_dtype)
            else:
                cond_1 = text_enc_1(input_ids_1, cond_mask_1)  # B L D
            cond_1 = cond_1.reshape(B, N, L, -1)
            cond_mask_1 = cond_mask_1.reshape(B, N, L)
            if text_enc_2 is not None:
                B_, N_, L_ = input_ids_2.shape  # B 1 L
                input_ids_2 = input_ids_2.reshape(-1, L_)
                if args.random_data:
                    cond_2 = torch.rand(B, 1280, device=x.device, dtype=weight_dtype)
                else:
                    cond_2 = text_enc_2(input_ids_2, cond_mask_2)  # B D
                cond_2 = cond_2.reshape(B_, 1, -1)  # B 1 D
            else:
                cond_2 = None

            # Map input images to latent space + normalize latents
            if args.random_data:
                b, c, t, h, w = x.shape
                x = torch.rand(
                    b, ae_channel_config[args.ae], 
                    (t - 1) // ae_stride_t + 1, h // ae_stride_h, w // ae_stride_w, 
                    device=x.device, dtype=x.dtype)
            else:
                x = ae.encode(x)  # B C T H W
            # print(f'step: {step_}, rank: {accelerator.process_index}, after vae.encode, x: {x.shape}, dtype: {x.dtype}, mean: {x.mean()}, std: {x.std()}')
            
            # def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
            #     from examples.rec_video import array_to_video
            #     x = x.detach().cpu()
            #     x = torch.clamp(x, -1, 1)
            #     x = (x + 1) / 2
            #     x = x.permute(1, 2, 3, 0).numpy()
            #     x = (255*x).astype(np.uint8)
            #     array_to_video(x, fps=fps, output_file=output_file)
            #     return
            # videos = ae.decode(x)[0]
            # videos = videos.transpose(0, 1)
            # custom_to_video(videos.to(torch.float32), fps=24, output_file='tmp.mp4')
            # import sys;sys.exit()
            
        # print("rank {} | step {} | after encode".format(accelerator.process_index, step_))

        current_step_frame = x.shape[2]
        current_step_sp_state = get_sequence_parallel_state()
        if args.sp_size != 1:  # enable sp
            if current_step_frame == 1:  # but image do not need sp
                set_sequence_parallel_state(False)
            else:
                set_sequence_parallel_state(True)
        if get_sequence_parallel_state():
            x, cond_1, attn_mask, cond_mask_1, cond_2 = prepare_parallel_data(
                x, cond_1, attn_mask, cond_mask_1, cond_2
                )        
            # x            (b c t h w)   -gather0-> (sp*b c t h w)   -scatter2-> (sp*b c t//sp h w)
            # cond_1       (b sp l/sp d) -gather0-> (sp*b sp l/sp d) -scatter1-> (sp*b 1 l/sp d)
            # attn_mask    (b t*sp h w)  -gather0-> (sp*b t*sp h w)  -scatter1-> (sp*b t h w)
            # cond_mask_1  (b sp l)      -gather0-> (sp*b sp l)      -scatter1-> (sp*b 1 l)
            # cond_2       (b sp d)      -gather0-> (sp*b sp d)      -scatter1-> (sp*b 1 d)
            for iter in range(args.train_batch_size * args.sp_size // args.train_sp_batch_size):
                with accelerator.accumulate(model):
                    # x            (sp_bs*b c t//sp h w)
                    # cond_1       (sp_bs*b 1 l/sp d)
                    # attn_mask    (sp_bs*b t h w)
                    # cond_mask_1  (sp_bs*b 1 l)
                    # cond_2       (sp_bs*b 1 d)
                    st_idx = iter * args.train_sp_batch_size
                    ed_idx = (iter + 1) * args.train_sp_batch_size
                    model_kwargs = dict(
                        encoder_hidden_states=cond_1[st_idx: ed_idx],
                        attention_mask=attn_mask[st_idx: ed_idx],
                        encoder_attention_mask=cond_mask_1[st_idx: ed_idx], 
                        pooled_projections=cond_2[st_idx: ed_idx] if cond_2 is not None else None, 
                        )
                    run(step_, x[st_idx: ed_idx], model_kwargs, prof_)
        else:
            with accelerator.accumulate(model):
                # assert not torch.any(torch.isnan(x)), 'after vae'
                x = x.to(weight_dtype)
                model_kwargs = dict(
                    encoder_hidden_states=cond_1, attention_mask=attn_mask, 
                    encoder_attention_mask=cond_mask_1, 
                    pooled_projections=cond_2
                    )
                run(step_, x, model_kwargs, prof_)

        set_sequence_parallel_state(current_step_sp_state)  # in case the next step use sp, which need broadcast(timesteps)

        if progress_info.global_step >= args.max_train_steps:
            return True

        return False

    def train_one_epoch(prof_=None):
        # for epoch in range(first_epoch, args.num_train_epochs):
        progress_info.train_loss = 0.0
        if progress_info.global_step >= args.max_train_steps:
            return True
        for step, data_item in enumerate(train_dataloader):
            # print("rank {} | step {} | get data".format(accelerator.process_index, step))
            if train_one_step(step, data_item, prof_):
                break

            if step >= 2 and torch_npu is not None and npu_config is not None:
                npu_config.free_mm()

    if npu_config is not None and npu_config.on_npu and npu_config.profiling:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization
        )
        profile_output_path = f"/home/image_data/npu_profiling_t2v/{os.getenv('PROJECT_NAME', 'local')}"
        os.makedirs(profile_output_path, exist_ok=True)

        with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU, 
                    torch_npu.profiler.ProfilerActivity.NPU, 
                    ],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(
                    wait=npu_config.profiling_step, warmup=0, active=1, repeat=1, skip_first=0
                    ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
        ) as prof:
            train_one_epoch(prof)
    else:
        if args.enable_profiling:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA, 
                    ], 
                schedule=torch.profiler.schedule(wait=5, warmup=1, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./gpu_profiling_active_1_delmask_delbkmask_andvaemask_curope_gpu'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                train_one_epoch(prof)
        else:
            train_one_epoch()


    
    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}")
        # FIXME: https://github.com/huggingface/accelerate/issues/3140
        acc_models = accelerator._models
        accelerator._models = [model for model in acc_models if model.checkpoint_engine is not None]
        accelerator.save_state(save_path)
        accelerator._models = acc_models

        model_path = os.path.join(save_path, "model")
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, 'config.json'), "w", encoding="utf-8") as writer:
            writer.write(model_config_to_save)
        convert_zero_checkpoint_to_fp32_state_dict(save_path, os.path.join(model_path, 'diffusion_pytorch_model.safetensors'))

        if args.use_ema:
            ema_model.save_pretrained(os.path.join(save_path, "model_ema"))
        logger.info(f"Saved state to {save_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    # =======================================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # deepspeed
    parser.add_argument("--train_deepspeed_config_file", type=str, required=True, help="deepspeed config file for training, e.g diffusion model")
    parser.add_argument("--eval_deepspeed_config_file", type=str, required=True, help="deepspeed config file for evaluation, e.g EMA model")

    # dataset & dataloader
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data", type=str, required='')
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--max_hxw", type=int, default=None)
    parser.add_argument("--min_hxw", type=int, default=None)
    parser.add_argument("--ood_img_ratio", type=float, default=0.0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--group_data", action="store_true")
    parser.add_argument("--hw_stride", type=int, default=32)
    parser.add_argument("--force_resolution", action="store_true")
    parser.add_argument("--force_5_ratio", action="store_true")
    parser.add_argument("--trained_data_global_step", type=int, default=None)
    parser.add_argument("--use_decord", action="store_true")
    parser.add_argument('--random_data', action='store_true')
    parser.add_argument('--force_zero_grad_step', type=int, default=-1)

    # text encoder & vae & diffusion model
    parser.add_argument('--vae_fp32', action='store_true')
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="Latte-XL/122")
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--interpolation_scale_h', type=float, default=1.0)
    parser.add_argument('--interpolation_scale_w', type=float, default=1.0)
    parser.add_argument('--interpolation_scale_t', type=float, default=1.0)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--text_encoder_name_1", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument('--sparse1d', action='store_true')
    parser.add_argument('--sparse_n', type=int, default=2)
    parser.add_argument('--skip_connection', action='store_true')
    parser.add_argument('--cogvideox_scheduler', action='store_true')
    parser.add_argument('--v1_5_scheduler', action='store_true')
    parser.add_argument('--rf_scheduler', action='store_true')
    parser.add_argument('--skip_abnorml_step', action='store_true')
    parser.add_argument("--post_to_device", action="store_true")
    parser.add_argument("--ema_bf16", action="store_true")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # diffusion setting
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--ema_update_freq", type=int, default=100)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument("--noise_offset", type=float, default=0.0, help="The scale of noise offset.")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument('--rescale_betas_zero_snr', action='store_true')
    parser.add_argument("--ema_decay_grad_clipping", type=float, default=0.9999)

    # validation & logs
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=4.5)
    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--proj_name", type=str, default=None, help="Custom project names for the runs in W&B logger, default to output_dir.")
    parser.add_argument("--log_name", type=str, default=None, help="Custom run names for the runs in W&B logger, default to proj_name or output_dir.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
                        )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
                        )
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    
    # optimizer & scheduler
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=1000000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimizer", type=str, default="adamW", help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-01, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True, help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True, help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_beta3", type=float, default=None,
                        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
                             "uses the value of square root of beta2. Ignored if optimizer is adamW",
                        )
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
                        ),
                        )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")

    args = parser.parse_args()
    main(args)
