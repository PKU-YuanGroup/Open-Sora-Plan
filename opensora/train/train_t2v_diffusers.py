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
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from packaging import version
from tqdm.auto import tqdm

import copy
import diffusers
from diffusers import DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
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
from opensora.utils.utils import explicit_uniform_sampling
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper

# from opensora.utils.utils import monitor_npu_power

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)

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
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch_npu is not None and npu_config is not None:
        npu_config.print_msg(args)
        npu_config.seed_everything(args.seed)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.num_frames != 1:
        initialize_sequence_parallel_state(args.sp_size)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

        # if accelerator.is_main_process:
        #     from threading import Thread
        #     Thread(target=monitor_npu_power, daemon=True).start()

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

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Create model:
    
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        kwargs = {}
        ae = ae_wrapper[args.ae](args.ae_path, cache_dir=args.cache_dir, **kwargs).eval()
        
        if args.enable_tiling:
            ae.vae.enable_tiling()

        kwargs = {
            'torch_dtype': weight_dtype, 
            'low_cpu_mem_usage': False
            }
        text_enc_1 = get_text_warpper(args.text_encoder_name_1)(args, **kwargs).eval()

        text_enc_2 = None
        if args.text_encoder_name_2 is not None:
            text_enc_2 = get_text_warpper(args.text_encoder_name_2)(args, **kwargs).eval()

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

    # # use pretrained model?
    if args.pretrained:
        model_state_dict = model.state_dict()
        print(f'Load from {args.pretrained}')
        if args.pretrained.endswith('.safetensors'):  
            from safetensors.torch import load_file as safe_load
            pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        elif os.path.isdir(args.pretrained):
            model = Diffusion_models_class[args.model].from_pretrained(args.pretrained)
            missing_keys, unexpected_keys = [], []
        else:
            pretrained_checkpoint = torch.load(args.pretrained, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']
                pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')

    model.gradient_checkpointing = args.gradient_checkpointing
    # Freeze vae and text encoders.
    ae.vae.requires_grad_(False)
    text_enc_1.requires_grad_(False)
    if text_enc_2 is not None:
        text_enc_2.requires_grad_(False)
    # Set model as trainable.
    model.train()

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
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if not args.extra_save_mem:
        ae.vae.to(accelerator.device, dtype=torch.float32 if args.vae_fp32 else weight_dtype)
        text_enc_1.to(accelerator.device, dtype=weight_dtype)
        if text_enc_2 is not None:
            text_enc_2.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_model = deepcopy(model)
        ema_model = EMAModel(ema_model.parameters(), decay=args.ema_decay, update_after_step=args.ema_start_step,
                             model_cls=Diffusion_models_class[args.model], model_config=ema_model.config, 
                             foreach=args.foreach_ema)

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
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "model_ema"), 
                    Diffusion_models_class[args.model], 
                    foreach=args.foreach_ema, 
                    )
                ema_model.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_model.pin_memory()
                else:
                    ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = Diffusion_models_class[args.model].from_pretrained(input_dir, subfolder="model")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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

    # Setup data:
    if args.trained_data_global_step is not None:
        initial_global_step_for_sampler = args.trained_data_global_step
    else:
        initial_global_step_for_sampler = 0
    

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
        pin_memory=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler, 
        drop_last=True, 
        # prefetch_factor=4
    )
    logger.info(f'after train_dataloader')

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
    # model.requires_grad_(False)
    # model.pos_embed.requires_grad_(True)
    # model.patch_embed.requires_grad_(True)

    logger.info(f'before accelerator.prepare')
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info(f'after accelerator.prepare')
    
    if args.use_ema:
        if args.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(os.path.basename(args.output_dir), config=vars(args))

    # Train!
    print(f"  Args = {args}")
    print(f"  noise_scheduler = {noise_scheduler}")
    logger.info("***** Running training *****")
    logger.info(f"  Model = {model}")
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
    progress_info = ProgressInfo(global_step, train_loss=0.0)

    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def sync_gradients_info(loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if args.use_ema:
            if args.offload_ema:
                ema_model.to(device="cuda", non_blocking=True)
            ema_model.step(model.parameters())
            if args.offload_ema:
                ema_model.to(device="cpu", non_blocking=True)
        progress_bar.update(1)
        progress_info.global_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        if progress_info.global_step % args.log_interval == 0:
            train_loss = progress_info.train_loss.item() / args.log_interval
            accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=progress_info.global_step)
            if torch_npu is not None and npu_config is not None:
                npu_config.print_msg(f"Step: [{progress_info.global_step}], local_loss={loss.detach().item()}, "
                                    f"train_loss={train_loss}, time_cost={one_step_duration}",
                                    rank=0)
            progress_info.train_loss = torch.tensor(0.0, device=loss.device)

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
                accelerator.save_state(save_path)
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
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        # avg_loss = accelerator.reduce(loss, reduction="mean")
        # progress_info.train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps
        progress_info.train_loss += avg_loss.detach() / args.gradient_accumulation_steps
        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params_to_clip = model.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
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
        # print("rank {} | step {} | unzip data".format(accelerator.process_index, step_))
        x, attn_mask, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2 = data_item_
        # print(f'step: {step_}, rank: {accelerator.process_index}, x: {x.shape}, dtype: {x.dtype}')
        # assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
        if args.extra_save_mem:
            torch.cuda.empty_cache()
            ae.vae.to(accelerator.device, dtype=torch.float32 if args.vae_fp32 else weight_dtype)
            text_enc_1.to(accelerator.device, dtype=weight_dtype)
            if text_enc_2 is not None:
                text_enc_2.to(accelerator.device, dtype=weight_dtype)

        x = x.to(accelerator.device, dtype=ae.vae.dtype)  # B C T H W
        # x = x.to(accelerator.device, dtype=torch.float32)  # B C T H W
        attn_mask = attn_mask.to(accelerator.device)  # B T H W
        input_ids_1 = input_ids_1.to(accelerator.device)  # B 1 L
        cond_mask_1 = cond_mask_1.to(accelerator.device)  # B 1 L
        input_ids_2 = input_ids_2.to(accelerator.device) if input_ids_2 is not None else input_ids_2 # B 1 L
        cond_mask_2 = cond_mask_2.to(accelerator.device) if cond_mask_2 is not None else cond_mask_2 # B 1 L
        
        with torch.no_grad():
            B, N, L = input_ids_1.shape  # B 1 L
            # use batch inference
            input_ids_1 = input_ids_1.reshape(-1, L)
            cond_mask_1 = cond_mask_1.reshape(-1, L)
            cond_1 = text_enc_1(input_ids_1, cond_mask_1)  # B L D
            cond_1 = cond_1.reshape(B, N, L, -1)
            cond_mask_1 = cond_mask_1.reshape(B, N, L)
            if text_enc_2 is not None:
                B_, N_, L_ = input_ids_2.shape  # B 1 L
                input_ids_2 = input_ids_2.reshape(-1, L_)
                cond_2 = text_enc_2(input_ids_2, cond_mask_2)  # B D
                cond_2 = cond_2.reshape(B_, 1, -1)  # B 1 D
            else:
                cond_2 = None

            # Map input images to latent space + normalize latents
            x = ae.encode(x)  # B C T H W
            # print(f'step: {step_}, rank: {accelerator.process_index}, after vae.encode, x: {x.shape}, dtype: {x.dtype}, mean: {x.mean()}, std: {x.std()}')
            # x = torch.rand(1, 32, 14, 80, 80).to(x.device, dtype=x.dtype)
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
        if args.extra_save_mem:
            ae.vae.to('cpu')
            text_enc_1.to('cpu')
            if text_enc_2 is not None:
                text_enc_2.to('cpu')
            torch.cuda.empty_cache()

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
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--trained_data_global_step", type=int, default=None)
    parser.add_argument("--use_decord", action="store_true")

    # text encoder & vae & diffusion model
    parser.add_argument('--vae_fp32', action='store_true')
    parser.add_argument('--extra_save_mem', action='store_true')
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
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # diffusion setting
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument("--noise_offset", type=float, default=0.0, help="The scale of noise offset.")
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument('--rescale_betas_zero_snr', action='store_true')

    # validation & logs
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=4.5)
    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
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
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimizer", type=str, default="adamW", help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-15, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
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
