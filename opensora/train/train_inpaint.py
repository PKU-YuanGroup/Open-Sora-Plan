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
import time
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
from torch.nn import functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available


from opensora.utils.ema import EMAModel
from opensora.dataset import getdataset
from opensora.models import CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc, get_text_warpper
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
from opensora.utils.dataset_utils import Collate, LengthGroupedSampler


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)

class ProgressInfo:
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # use LayerNorm, GeLu, SiLu always as fp32 mode
    if args.enable_stable_fp32:
        replace_with_fp32_forwards()
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

    if args.num_frames != 1 and args.use_image_num == 0:
        initialize_sequence_parallel_state(args.sp_size)

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

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Create model:
    kwargs = {}
    ae = CausalVAEModelWrapper(args.ae_path, cache_dir=args.cache_dir, **kwargs).eval()
    if args.enable_tiling:
        ae.vae.enable_tiling()
        ae.vae.tile_overlap_factor = args.tile_overlap_factor

    kwargs = {'load_in_8bit': args.enable_8bit_t5, 'torch_dtype': weight_dtype, 'low_cpu_mem_usage': True}
    text_enc = get_text_warpper(args.text_encoder_name)(args, **kwargs).eval()

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
    # assert args.num_frames % ae_stride_t == 0, f"Num_frames must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert args.max_height % ae_stride_h == 0, f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
    assert args.max_width % ae_stride_h == 0, f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    ae.latent_size = latent_size

    if args.num_frames % 2 == 1:
        args.latent_size_t = latent_size_t = (args.num_frames - 1) // ae_stride_t + 1
    else:
        latent_size_t = args.num_frames // ae_stride_t

    model_kwargs = {'vae_scale_factor_t': ae_stride_t}

    model = Diffusion_models[args.model](
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae],
        # caption_channels=4096,
        # cross_attention_dim=1152,
        attention_bias=True,
        sample_size=latent_size,
        sample_size_t=latent_size_t,
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
        attention_mode=args.attention_mode,
        interpolation_scale_h=args.interpolation_scale_h,
        interpolation_scale_w=args.interpolation_scale_w,
        interpolation_scale_t=args.interpolation_scale_t,
        downsampler=args.downsampler,
        # compress_kv_factor=args.compress_kv_factor,
        use_rope=args.use_rope,
        # model_max_length=args.model_max_length,
        use_stable_fp32=args.enable_stable_fp32,
        **model_kwargs,
    )
    model.gradient_checkpointing = args.gradient_checkpointing

    pretrained_transformer_model_path = args.pretrained_transformer_model_path
    pretrained_model_path = dict(transformer_model=pretrained_transformer_model_path)
    if pretrained_transformer_model_path is not None:
        model.custom_load_state_dict(pretrained_model_path)

    noise_scheduler = DDPMScheduler()

    # Freeze main models
    ae.vae.requires_grad_(False)
    text_enc.requires_grad_(False)

    model.train()

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    ae.vae.to(accelerator.device, dtype=torch.float32)
    # ae.vae.to(accelerator.device, dtype=weight_dtype)
    text_enc.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_model = deepcopy(model)
        ema_model = EMAModel(ema_model.parameters(), decay=args.ema_decay, update_after_step=args.ema_start_step,
                             model_cls=Diffusion_models_class[args.model], model_config=ema_model.config)
    
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
            # loading ema with customed 'from_pretrained' function
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), Diffusion_models_class[args.model])
                ema_model.load_state_dict(load_model.state_dict())
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

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
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
    train_dataset = getdataset(args)
    logger.info(f"train_dataset: {train_dataset.__class__.__name__}")
    sampler = LengthGroupedSampler(
                args.train_batch_size,
                world_size=accelerator.num_processes,
                lengths=train_dataset.lengths, 
                group_frame=args.group_frame, 
                group_resolution=args.group_resolution, 
            ) if args.group_frame or args.group_resolution else None
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        # pin_memory=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler if args.group_frame or args.group_resolution else None, 
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

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # NOTE wandb
    if accelerator.is_main_process:
        logger.info("init trackers...")
        project_name = os.getenv('PROJECT', os.path.basename(args.output_dir))
        entity = os.getenv('ENTITY', None)
        run_name = os.getenv('WANDB_NAME', None)
        init_kwargs = {
            "entity": entity,
            "run_name": run_name,
        }
        accelerator.init_trackers(project_name=project_name, config=vars(args), init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
    logger.info("***** Running training *****")
    logger.info(f"  Model = {model}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
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

            if npu_config is not None:
                train_dataset.n_used_elements = global_step * args.train_batch_size

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

    def sync_gradients_info(loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if args.use_ema:
            ema_model.step(model.parameters())
        progress_bar.update(1)
        progress_info.global_step += 1
        end_time = time.time()
        one_step_duration = end_time - start_time
        accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
        if torch_npu is not None and npu_config is not None:
            npu_config.print_msg(f"Step: [{progress_info.global_step}], local_loss={loss.detach().item()}, "
                                 f"train_loss={progress_info.train_loss}, time_cost={one_step_duration}",
                                 rank=0)
        progress_info.train_loss = 0.0

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

    def run(model_input, model_kwargs, prof):
        global start_time
        start_time = time.time()

        try:
            in_channels = ae_channel_config[args.ae]
            model_input, masked_x, video_mask = model_input[:, 0:in_channels], model_input[:, in_channels:2 * in_channels], model_input[:, 2 * in_channels:]
        except:
            raise ValueError("masked_x and video_mask is None!")

        noise = torch.randn_like(model_input)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn((model_input.shape[0], model_input.shape[1], 1, 1, 1),
                                                     device=model_input.device)

        bsz = model_input.shape[0]
        current_step_frame = model_input.shape[2]
        # Sample a random timestep for each image without bias.
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        if current_step_frame != 1 and get_sequence_parallel_state():  # image do not need sp
            broadcast(timesteps)

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = model(
            torch.cat([noisy_model_input, masked_x, video_mask], dim=1),
            timesteps,
            **model_kwargs,
        )[0]
       
        # Get the target for loss depending on the prediction type
        if args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            noise_scheduler.register_to_config(prediction_type=args.prediction_type)

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


        mask = model_kwargs.get('attention_mask', None)
        if torch.all(mask.bool()):
            mask = None
        if get_sequence_parallel_state():
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
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(b, -1)
            mse_loss_weights = mse_loss_weights.reshape(b, 1)
            if mask is not None:
                loss = (loss * mask * mse_loss_weights).sum() / mask.sum()  # mean loss on unpad patches
            else:
                loss = (loss * mse_loss_weights).mean()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        progress_info.train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params_to_clip = params_to_optimize
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            sync_gradients_info(loss)

        if prof is not None:
            prof.step()


        return loss

    def train_one_step(step_, data_item_, prof_=None):
        train_loss = 0.0
        x, attn_mask, input_ids, cond_mask = data_item_
        # assert torch.all(attn_mask.bool()), 'must all visible'
        # Sample noise that we'll add to the latents
        # import ipdb;ipdb.set_trace()
        if args.group_frame or args.group_resolution:
            if not args.group_frame:
                each_latent_frame = torch.any(attn_mask.flatten(-2), dim=-1).int().sum(-1).tolist()
                # logger.info(f'rank: {accelerator.process_index}, step {step_}, special batch has attention_mask '
                #             f'each_latent_frame: {each_latent_frame}')
                logger.info(f'rank: {accelerator.process_index}, step {step_}, special batch has attention_mask '
                            f'each_latent_frame: {each_latent_frame}')
        assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
        x = x.to(accelerator.device, dtype=ae.vae.dtype)  # B 3*C T H W, 16

        attn_mask = attn_mask.to(accelerator.device)  # B T H W
        input_ids = input_ids.to(accelerator.device)  # B 1 L
        cond_mask = cond_mask.to(accelerator.device)  # B 1 L

        with torch.no_grad():
            B, N, L = input_ids.shape  # B 1 L
            # use batch inference
            input_ids_ = input_ids.reshape(-1, L)
            cond_mask_ = cond_mask.reshape(-1, L)
            cond = text_enc(input_ids_, cond_mask_)  # B 1 L D
            cond = cond.reshape(B, N, L, -1)

            def preprocess_x_for_inpaint(x):
                # NOTE vae-styled mask, deprecated
                if args.use_vae_preprocessed_mask:
                    x, masked_x, mask = x[:, :3], x[:, 3:6], x[:, 6:9]
                    x, masked_x, mask = ae.encode(x), ae.encode(masked_x), ae.encode(mask)
                else:
                    x, masked_x, mask = x[:, :3], x[:, 3:6], x[:, 6:7]
                    x, masked_x = ae.encode(x), ae.encode(masked_x)
                    batch_size, channels, frame, height, width = mask.shape
                    mask = rearrange(mask, 'b c t h w -> (b c t) 1 h w')
                    mask = F.interpolate(mask, size=latent_size, mode='bilinear')
                    mask = rearrange(mask, '(b c t) 1 h w -> b c t h w', t=frame, b=batch_size)
                    mask_first_frame = mask[:, :, 0:1].repeat(1, 1, ae_stride_t, 1, 1).contiguous()
                    mask = torch.cat([mask_first_frame, mask[:, :, 1:]], dim=2)
                    mask = mask.view(batch_size, ae_stride_t, latent_size_t, latent_size[0], latent_size[1]).contiguous()

                return x, masked_x, mask

            # Map input images to latent space + normalize latents
            x, masked_x, mask = preprocess_x_for_inpaint(x) # B 3*C T H W -> (B C T H W) * 3 
            x = torch.cat([x, masked_x, mask], dim=1) # (B C T H W) * 3 -> B 3*C T H W

        current_step_frame = x.shape[2]
        current_step_sp_state = get_sequence_parallel_state()
        if args.sp_size != 1:  # enable sp
            if current_step_frame == 1:  # but image do not need sp
                set_sequence_parallel_state(False)
            else:
                set_sequence_parallel_state(True)
        if get_sequence_parallel_state():
            x, cond, attn_mask, cond_mask, use_image_num = prepare_parallel_data(x, cond, attn_mask, cond_mask,
                                                                                 args.use_image_num)
            for iter in range(args.train_batch_size * args.sp_size // args.train_sp_batch_size):
                with accelerator.accumulate(model):
                    st_idx = iter * args.train_sp_batch_size
                    ed_idx = (iter + 1) * args.train_sp_batch_size
                    model_kwargs = dict(encoder_hidden_states=cond[st_idx: ed_idx],
                                        attention_mask=attn_mask[st_idx: ed_idx],
                                        encoder_attention_mask=cond_mask[st_idx: ed_idx], use_image_num=use_image_num)
                    run(x[st_idx: ed_idx], model_kwargs, prof_)

        else:
            with accelerator.accumulate(model):
                assert not torch.any(torch.isnan(x)), 'after vae'
                x = x.to(weight_dtype)
                model_kwargs = dict(encoder_hidden_states=cond, attention_mask=attn_mask,
                                    encoder_attention_mask=cond_mask, use_image_num=args.use_image_num,)
                run(x, model_kwargs, prof_)

        set_sequence_parallel_state(current_step_sp_state)  # in case the next step use sp, which need broadcast(timesteps)

        if progress_info.global_step >= args.max_train_steps:
            return True

        return False

    def train_all_epoch(prof_=None):
        for epoch in range(first_epoch, args.num_train_epochs):
            progress_info.train_loss = 0.0
            if progress_info.global_step >= args.max_train_steps:
                return True

            for step, data_item in enumerate(train_dataloader):
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
                activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=npu_config.profiling_step, warmup=0, active=1, repeat=1,
                                                     skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{profile_output_path}/")
        ) as prof:
            train_all_epoch(prof)
    else:
        train_all_epoch()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if npu_config is not None and get_sequence_parallel_state():
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
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")

    # text encoder & vae & diffusion model
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="Latte-XL/122")
    parser.add_argument('--enable_8bit_t5', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.125)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument("--attention_mode", type=str, choices=['xformers', 'math', 'flash'], default="xformers")
    parser.add_argument('--use_rope', action='store_true')
    parser.add_argument('--compress_kv_factor', type=int, default=1)
    parser.add_argument('--interpolation_scale_h', type=float, default=1.0)
    parser.add_argument('--interpolation_scale_w', type=float, default=1.0)
    parser.add_argument('--interpolation_scale_t', type=float, default=1.0)
    parser.add_argument("--downsampler", type=str, default=None)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument('--enable_stable_fp32', action='store_true')
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # diffusion setting
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--noise_offset", type=float, default=0.02, help="The scale of noise offset.")
    parser.add_argument("--prediction_type", type=str, default=None, help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")

    # validation & logs
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=2.5)
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
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
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

    # inpaint
    parser.add_argument("--i2v_ratio", type=float, default=0.5) # for inpainting mode
    parser.add_argument("--transition_ratio", type=float, default=0.4) # for inpainting mode
    parser.add_argument("--v2v_ratio", type=float, default=0.1) # for inpainting mode
    parser.add_argument("--clear_video_ratio", type=float, default=0.0)
    parser.add_argument("--default_text_ratio", type=float, default=0.1)
    parser.add_argument("--pretrained_transformer_model_path", type=str, default=None)
    parser.add_argument("--use_vae_preprocessed_mask", action="store_true")

    args = parser.parse_args()
    main(args)