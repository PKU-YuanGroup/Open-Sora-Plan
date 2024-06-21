# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test NaViT output consistency with the original implementation.
"""
import argparse
import logging
import math
import os
import shutil
from copy import deepcopy
from logging import getLogger
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from einops import rearrange
from huggingface_hub import create_repo
from packaging import version
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from opensora.dataset import ae_denorm, getdataset
from opensora.models.ae import ae_channel_config, ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVAEModelWrapper, CausalVQVAEModelWrapper
from opensora.models.diffusion import Diffusion_models
from opensora.utils.dataset_utils import NaViTCollate
from opensora.models.diffusion.diffusion import (
    create_diffusion_navit as create_diffusion,
)
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.diffusion.latte.modeling_latte_navit import (
    NaViTLatteT2V,
    pack_target_as,
)
from opensora.models.text_encoder import get_text_enc

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = getLogger(__name__)


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create model:

    diffusion = create_diffusion(
        timestep_respacing="",
    )  # default: 1000 steps, linear noise schedule
    ae = getae_wrapper(args.ae)(
        args.ae_path, subfolder="vae", cache_dir="cache_dir"
    ).eval()
    if args.enable_tiling:
        ae.vae.enable_tiling()
        ae.vae.tile_overlap_factor = args.tile_overlap_factor
    text_enc = get_text_enc(args).eval()

    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = (
        ae_stride_t,
        ae_stride_h,
        ae_stride_w,
    )
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = (
        int(patch_size[0]),
        int(patch_size[1]),
        int(patch_size[2]),
    )
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = (
        patch_size_t,
        patch_size_h,
        patch_size_w,
    )
    assert (
        ae_stride_h == ae_stride_w
    ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    assert (
        patch_size_h == patch_size_w
    ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    # assert args.num_frames % ae_stride_t == 0, f"Num_frames must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert (
        args.max_image_size % ae_stride_h == 0
    ), f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}),  ae_stride_h ({ae_stride_h})."

    latent_size = (
        args.max_image_size // ae_stride_h,
        args.max_image_size // ae_stride_w,
    )

    if (
        getae_wrapper(args.ae) == CausalVQVAEModelWrapper
        or getae_wrapper(args.ae) == CausalVAEModelWrapper
    ):
        args.video_length = video_length = args.num_frames // ae_stride_t + 1
    else:
        video_length = args.num_frames // ae_stride_t

    model = Diffusion_models["LatteT2V-XL/122"](
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
        attention_type="default",
        video_length=video_length,
        attention_mode=args.attention_mode,
    )
    new_model: torch.nn.Module = Diffusion_models["NaViTLatteT2V-XL/122"](
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
        attention_type="default",
        video_length=video_length,
        attention_mode=args.attention_mode,
        max_token_lim=args.max_token_lim,
        token_dropout_rate=args.token_dropout_rate,
    )

    # # use pretrained model?
    if args.pretrained:
        if "safetensors" in args.pretrained:
            from safetensors.torch import load_file as safe_load

            checkpoint = safe_load(args.pretrained, device="cpu")
        else:
            checkpoint = torch.load(args.pretrained, map_location="cpu")[
                "model"
            ]
        model_state_dict = model.state_dict()
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint, strict=False
        )
        logger.info(
            f"missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}"
        )
        logger.info(
            f"Successfully load {len(model.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!"
        )
        # load from pixart-alpha
        # pixelart_alpha = torch.load(args.pretrained, map_location='cpu')['state_dict']
        # checkpoint = {}
        # for k, v in pixelart_alpha.items():
        #     if 'x_embedder' in k or 't_embedder' in k or 'y_embedder' in k:
        #         checkpoint[k] = v
        #     if k.startswith('blocks'):
        #         k_spilt = k.split('.')
        #         blk_id = str(int(k_spilt[1]) * 2)
        #         k_spilt[1] = blk_id
        #         new_k = '.'.join(k_spilt)
        #         checkpoint[new_k] = v
        # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        # logger.info(f'Successfully load {len(model.state_dict()) - len(missing_keys)} keys from {args.pretrained}!')

    # copy weight from model to new_model
    logger.info("Copying weight from model to new_model...")
    model_state_dict = model.state_dict()
    missing_keys, unexpected_keys = new_model.load_state_dict(
        model_state_dict, strict=True
    )
    logger.info(
        f"missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}"
    )
    logger.info(
        f"Successfully load {len(model.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys!"
    )

    # Freeze vae and text encoders.
    ae.requires_grad_(False)
    text_enc.requires_grad_(False)
    # Set model as eval.
    model.train()
    new_model.train()

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # weight_dtype = torch.bfloat16
    weight_dtype = torch.float32

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    device = torch.device("cuda:0")
    ae.to(device, dtype=torch.float32)
    text_enc.to(device, dtype=weight_dtype)
    model.to(device, dtype=weight_dtype)
    new_model.to(device, dtype=weight_dtype)


    # Setup data:
    train_dataset = getdataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=NaViTCollate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    for step, (x_list, input_ids, cond_mask) in enumerate(train_dataloader):
        x = torch.stack(x_list)
        x = x.to(device)

        
        attn_mask = None
        input_ids = input_ids.to(device)  # B L
        cond_mask = cond_mask.to(device)  # B L

        with torch.no_grad():
            # Map input images to latent space + normalize latents
            B, _, _ = input_ids.shape  # B T+num_images L  b 1+4, L
            cond = torch.stack([text_enc(input_ids[i], cond_mask[i]) for i in range(B)])  # B 1+num_images L D
            if args.use_image_num == 0:
                x = ae.encode(x)  # B C T H W
            else:
                videos, images = x[:, :, :-args.use_image_num], x[:, :, -args.use_image_num:]
                videos = ae.encode(videos)  # B C T H W

                images = rearrange(images, 'b c t h w -> (b t) c 1 h w')
                images = ae.encode(images)

                images = rearrange(images, '(b t) c 1 h w -> b c t h w', t=args.use_image_num)
                x = torch.cat([videos, images], dim=2)   #  b c 17+4, h, w

            model_kwargs = dict(
                    encoder_hidden_states=cond,
                    attention_mask=attn_mask,
                    encoder_attention_mask=cond_mask,
                    use_image_num=args.use_image_num,
                )
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # [B, C, F, H, W]
            output = model(x, t, **model_kwargs)[0]

            # delete model to save memory
            # del model


            # new model branch
            x_list = list(x.unbind(dim=0))
            # delete x to save memory
            del x

            torch.cuda.empty_cache()
            

            # [b, F, T, p*p*c]
            new_output, video_ids, token_kept_ids = new_model(x_list, t, **model_kwargs)
            output = list(output.unbind(dim=0))
            # [b, F, T, p*p*c]
            output = pack_target_as(output, video_ids, new_model.patch_size, token_kept_ids)

            diff_max = torch.abs(output - new_output) > 1e-5

            print("output: ", output[diff_max], "\nnew_output: ", new_output[diff_max])

            torch.testing.assert_close(output, new_output)
            print("all matched!")




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
    parser.add_argument(
        "--max_token_lim",
        type=int,
        default=1024,
        help="The max token limit of NaViT training.",
    )
    parser.add_argument(
        "--token_dropout_rate",
        type=float,
        default=0,
        help="The drop rate of token in NaViT training.",
    )

    args = parser.parse_args()
    main(args)