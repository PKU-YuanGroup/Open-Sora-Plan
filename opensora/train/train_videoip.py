# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
from email.mime import image
import logging
import math
from mimetypes import init
import os
from selectors import EpollSelector
import shutil
from pathlib import Path
from typing import Optional
import gc
import numpy as np
from einops import rearrange
from tqdm import tqdm
import itertools

from opensora.models.ae.videobase.modules import attention

import time
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer
from math import sqrt

import diffusers
from diffusers import DDPMScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from opensora.dataset import getdataset, ae_denorm
from opensora.models.ae import getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.text_encoder import get_text_enc, get_text_warpper
from opensora.utils.dataset_utils import VideoIP_Collate as Collate
from opensora.models.ae import ae_stride_config, ae_channel_config
from opensora.models.diffusion import Diffusion_models, Diffusion_models_class
from opensora.sample.pipeline_opensora import OpenSoraPipeline


# for validation
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda
from transformers import CLIPVisionModelWithProjection, AutoModel, AutoImageProcessor, CLIPImageProcessor
from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop, LongSideResizeVideo, SpatialStrideCropVideo
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.sample.pipeline_for_vip import hacked_pipeline_call_for_vip

from opensora.models.diffusion.latte.videoip import VideoIPAdapter, VideoIPAttnProcessor
from opensora.models.diffusion.latte.modeling_for_vip import hacked_forward_for_vip, hook_forward_fn, hook_backward_fn
from opensora.models.diffusion.latte.modules import BasicTransformerBlock

import glob
from torchvision.utils import save_image
import imageio

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)


class VIPNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        image_encoder_type='dino',
        cross_attention_dim=1152,
        num_tokens=272,
        vip_inner_dim=1024,
        vip_num_attention_layers=2,
        attention_mode='xformers',
        gradient_checkpointing=False,
        vae_scale_factor_t=4,
        video_length=17,
        use_rope=False,
        rope_scaling=None,
        attn_proc_type_dict={},
    ):
        super().__init__()

        self.image_encoder_type = image_encoder_type
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling


        self.vip_adapter = VideoIPAdapter(
            image_encoder_type=image_encoder_type,
            cross_attention_dim=cross_attention_dim,
            inner_dim=vip_inner_dim,
            num_attention_layers=vip_num_attention_layers,
            use_rope=use_rope,
            rope_scaling=rope_scaling,
            attention_mode=attention_mode,
            gradient_checkpointing=gradient_checkpointing,
            vae_scale_factor_t=vae_scale_factor_t,
            video_length=video_length,
        )


        self.attn_procs = {}
        # because nn.ModuleDict will raise (KeyError: 'module name can\'t contain ".", got: transformer_blocks.0.attn2.processor'), so we trun to use nn.ModuleList
        for name, attn_proc_type in attn_proc_type_dict.items():
            if attn_proc_type == "VideoIPAttnProcessor":
                self.attn_procs[name] = VideoIPAttnProcessor(
                    dim=cross_attention_dim,
                    attention_mode=attention_mode,
                    use_rope=use_rope,
                    rope_scaling=rope_scaling,
                    num_vip_tokens=num_tokens,
                )
        self.adapter_modules = torch.nn.ModuleList(self.attn_procs.values())

        # if pretrained_vip_adapter_path is not None:
        #     self.load_vip_adapter(pretrained_vip_adapter_path)

    def set_vip_adapter(self, model, init_from_original_attn_processor):
        # init adapter modules
        model_sd = model.state_dict()
        attn_procs = {}
        print("set vip adapter...")
        for name, attn_processor in model.attn_processors.items():
            if name.endswith(".attn2.processor"):
                new_attn_processor = self.attn_procs[name]
                if init_from_original_attn_processor: 
                    print("init from original attn processor...")
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_vip.weight": model_sd[layer_name + ".to_k.weight"],
                        "to_v_vip.weight": model_sd[layer_name + ".to_v.weight"],
                    }
                    new_attn_processor.load_state_dict(weights, strict=False)
                attn_procs[name] = new_attn_processor
            else:
                attn_procs[name] = attn_processor
                
        model.set_attn_processor(attn_procs)

    @torch.no_grad()
    def get_image_embeds(self, images, image_processor, image_encoder, transform, device, weight_dtype=torch.float32):
        if not isinstance(images, list):
            images = [images]
        images = [Image.open(image).convert("RGB") for image in images]
        images = [torch.from_numpy(np.copy(np.array(image))).unsqueeze(0) for image in images] # 1 H W C
        images = torch.cat([transform(image.permute(0, 3, 1, 2).float()).to(torch.uint8) for image in images]) # resize, 1 C H W

        images = image_processor(images=images, return_tensors="pt").pixel_values # 1 C H W
        images = images.to(device=device, dtype=weight_dtype)
        negative_images = torch.zeros_like(images, device=device, dtype=weight_dtype)

        images = image_encoder(images).last_hidden_state # 1 N D
        images = images[:, 1:] # drop cls token
        negative_images = image_encoder(negative_images).last_hidden_state
        negative_images = negative_images[:, 1:]

        height = width = int(sqrt(images.shape[1]))
        images = rearrange(images, '1 (h w) c -> c 1 h w', h=height, w=width)
        negative_images = rearrange(negative_images, '1 (h w) c -> c 1 h w', h=height, w=width)
        images = images.unsqueeze(0) # 1 C 1 H W
        negative_images = negative_images.unsqueeze(0)

        vip_out = self.vip_adapter(hidden_states=images, use_image_num=0) 
        vip_tokens, vip_cond_mask = vip_out.hidden_states, vip_out.vip_cond_mask # 1 1 N D, 1 1 N

        negative_vip_out = self.vip_adapter(hidden_states=negative_images, use_image_num=0)
        negative_vip_tokens, negative_vip_cond_mask = negative_vip_out.hidden_states, negative_vip_out.vip_cond_mask

        return vip_tokens, vip_cond_mask, negative_vip_tokens, negative_vip_cond_mask

    @torch.no_grad()
    def get_video_embeds(self, condition_images, num_frames, image_processor, image_encoder, transform, device, weight_dtype=torch.float32):
        if len(condition_images) == 1:
            condition_images_indices = [0]
        elif len(condition_images) == 2:
            condition_images_indices = [0, -1]
        condition_images = [Image.open(image).convert("RGB") for image in condition_images]
        condition_images = [torch.from_numpy(np.copy(np.array(image))).unsqueeze(0) for image in condition_images] # F [1 H W C]
        condition_images = torch.cat([transform(image.permute(0, 3, 1, 2).float()).to(torch.uint8) for image in condition_images]) # resize, [F C H W]

        condition_images = image_processor(images=condition_images, return_tensors="pt").pixel_values # F C H W
        condition_images = condition_images.to(device=device, dtype=weight_dtype)
        _, C, H, W = condition_images.shape
        video = torch.zeros([num_frames, C, H, W], device=device, dtype=weight_dtype)
        video[condition_images_indices] = condition_images
        negative_video = torch.zeros_like(video, device=device, dtype=weight_dtype)

        video = image_encoder(video).last_hidden_state # F N D
        video = video[:, 1:] # drop cls token
        negative_video = image_encoder(negative_video).last_hidden_state
        negative_video = negative_video[:, 1:]

        height = width = int(sqrt(video.shape[1]))
        video = rearrange(video, 't (h w) c -> c t h w', h=height, w=width)
        negative_video = rearrange(negative_video, 't (h w) c -> c t h w', h=height, w=width)

        video = video.unsqueeze(0) # 1 C F H W
        negative_video = negative_video.unsqueeze(0)

        vip_out = self.vip_adapter(hidden_states=video, use_image_num=0)
        vip_tokens, vip_cond_mask = vip_out.hidden_states, vip_out.vip_cond_mask # 1 1 N D, 1 1 N

        negative_vip_out = self.vip_adapter(hidden_states=negative_video, use_image_num=0)
        negative_vip_tokens, negative_vip_cond_mask = negative_vip_out.hidden_states, negative_vip_out.vip_cond_mask

        return vip_tokens, vip_cond_mask, negative_vip_tokens, negative_vip_cond_mask


    def forward(self, model, latent_model_input, timestep, **model_kwargs):
        enable_temporal_attentions = True if args.num_frames > 1 else False

        encoder_hidden_states = model_kwargs.pop('encoder_hidden_states', None)
        encoder_attention_mask = model_kwargs.pop('encoder_attention_mask', None)
        clip_feature = model_kwargs.pop('clip_feature', None)
        use_image_num = model_kwargs.pop('use_image_num', 0)
        assert encoder_hidden_states is not None and encoder_attention_mask is not None and clip_feature is not None, "VIPNet requires encoder_hidden_states, encoder_attention_mask and clip_feature"
        hidden_states = rearrange(clip_feature, 'b t h w c -> b c t h w')

        vip_out = self.vip_adapter(hidden_states=hidden_states, use_image_num=use_image_num) # B D T+image_num H W  -> B 1+image_num N D
        vip_tokens, vip_cond_mask = vip_out.hidden_states, vip_out.vip_cond_mask
        
        model_pred = model(
            hidden_states=latent_model_input, 
            timestep=timestep, 
            encoder_hidden_states=encoder_hidden_states, 
            encoder_attention_mask=encoder_attention_mask, 
            vip_hidden_states=vip_tokens,
            vip_attention_mask=vip_cond_mask,
            use_image_num=use_image_num,
            enable_temporal_attentions=enable_temporal_attentions,
            **model_kwargs
        )[0]

        return model_pred


@torch.inference_mode()
def log_validation(
    args,   
    model,
    vip,
    vae, 
    text_encoder, 
    tokenizer, 
    image_processor,
    image_encoder,
    accelerator, 
    weight_dtype, 
    global_step, 
    ema=False
):


    validation_dir = args.validation_dir if args.validation_dir is not None else "./validation"
    prompt_file = os.path.join(validation_dir, "prompt.txt")

    with open(prompt_file, 'r') as f:
        validation_prompt = f.readlines()

    index = 0
    validation_images_list = []
    while True:
        temp = glob.glob(os.path.join(validation_dir, f"*_{index:04d}*.png"))
        print(temp)
        if len(temp) > 0:
            validation_images_list.append(sorted(temp))
            index += 1
        else:
            break

    resize_transform = transforms.Compose(
        [CenterCropResizeVideo((args.max_height, args.max_width))]
    )
    logger.info(f"Running {'normal' if not ema else 'ema'} validation....\n")

    vip = accelerator.unwrap_model(vip)

    vip.eval()

    scheduler = PNDMScheduler()
    pipeline = OpenSoraPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=model
    ).to(device=accelerator.device)


    pipeline.__call__ = hacked_pipeline_call_for_vip.__get__(pipeline, OpenSoraPipeline)

    videos = []
    gen_img = False
    for prompt, images in zip(validation_prompt, validation_images_list):
        if not isinstance(images, list):
            images = [images]
        logger.info('Processing the ({}) prompt and the images ({})'.format(prompt, images))
        if args.num_frames == 1:
            if len(images) == 1 and images[0].split('/')[-1].split('_')[0] == 'img': 
                vip_tokens, vip_cond_mask, negative_vip_tokens, negative_vip_cond_mask = vip.get_image_embeds(
                    images=images, 
                    image_processor=image_processor,
                    image_encoder=image_encoder, 
                    transform=resize_transform,
                    device=accelerator.device,
                    weight_dtype=torch.float32
                )
                gen_img = True
            else:
                vip_tokens, vip_cond_mask, negative_vip_tokens, negative_vip_cond_mask = vip.get_video_embeds(
                    condition_images=images,
                    num_frames=args.num_frames,
                    image_processor=image_processor,
                    image_encoder=image_encoder,
                    transform=resize_transform,
                    device=accelerator.device,
                    weight_dtype=torch.float32
                )
        else:
            vip_tokens, vip_cond_mask, negative_vip_tokens, negative_vip_cond_mask = vip.get_image_embeds(
                images=images[0], # only using first image
                image_processor=image_processor,
                image_encoder=image_encoder, 
                transform=resize_transform,
                device=accelerator.device,
                weight_dtype=torch.float32
            )
            gen_img = True

        video = pipeline.__call__(
            prompt=prompt,
            negative_prompt="",
            vip_tokens=vip_tokens,
            vip_attention_mask=vip_cond_mask,
            negative_vip_tokens=negative_vip_tokens,
            negative_vip_attention_mask=negative_vip_cond_mask,
            num_frames=(1 if gen_img else args.num_frames),
            height=args.max_height,
            width=args.max_width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            mask_feature=True,
            device=accelerator.device,
            max_squence_length=args.model_max_length,
        ).images
        videos.append(video[0])
        gen_img = False
    # import ipdb;ipdb.set_trace()

    # Save the generated videos
    save_dir = os.path.join(args.output_dir, f"val_{global_step:09d}" if not ema else f"val_ema_{global_step:09d}")
    os.makedirs(save_dir, exist_ok=True)

    for idx, video in enumerate(videos):

        if video.shape[0] == 1: # image
            ext = 'png'
            Image.fromarray(video[0].cpu().numpy()).save(os.path.join(save_dir, f'{idx}.{ext}'))
        else: # video
            ext = 'mp4'
            imageio.mimwrite(
                os.path.join(save_dir, f'{idx}.{ext}'), video, fps=24, quality=6)  # highest quality is 10, lowest is 0
        

    # for wandb
    resize_transform = transforms.Compose(
        [CenterCropResizeVideo((args.max_height // 4, args.max_width // 4))]
    )

    videos = [resize_transform(video.permute(0, 3, 1, 2)) for video in videos]


    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            import wandb
            logs = {}
            logs[f"{'ema_' if ema else ''}validation_videos"] = []
            logs[f"{'ema_' if ema else ''}validation_images"] = []
            for i, (video, prompt) in enumerate(zip(videos, validation_prompt)):
                if video.shape[0] == 1: # image
                    logs[f"{'ema_' if ema else ''}validation_images"].append(wandb.Image(video[0], caption=f"{i}: {prompt}"))
                else: # video
                    logs[f"{'ema_' if ema else ''}validation_videos"].append(wandb.Video(video, caption=f"{i}: {prompt}", fps=24))

            tracker.log(logs, step=global_step)

    print("delete validation pipeline...")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    vip.train()

class ProgressInfo:
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss


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

    if accelerator.is_main_process:
        print("--------------------------training args--------------------------")
        print(args)
        print("-----------------------------------------------------------------")

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
    ae = getae_wrapper(args.ae)(args.ae_path, cache_dir=args.cache_dir, **kwargs).eval()
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
        args.latent_size_t = latent_size_t = args.num_frames // ae_stride_t + 1
    else:
        latent_size_t = args.num_frames // ae_stride_t
    # when training video ip adapter, we use t2v model
    model = Diffusion_models[args.model](
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae] * 2, # 因为要加载预训练权重，所以这里out_channels仍然设置为2倍
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
        downsampler=args.downsampler,
        compress_kv_factor=args.compress_kv_factor,
        use_rope=args.use_rope,
        model_max_length=args.model_max_length,
    )
    model.gradient_checkpointing = args.gradient_checkpointing

    # NOTE replace forward for VIP
    model.forward = hacked_forward_for_vip.__get__(model, Diffusion_models[args.model])

    # # use pretrained model?
    # NOTE when using inpaint model
    # if args.pretrained:
    #     model.custom_load_state_dict(args.pretrained)

    if args.pretrained:
        model_state_dict = model.state_dict()
        if 'safetensors' in args.pretrained:  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(args.pretrained, device="cpu")
            pretrained_keys = set(list(pretrained_checkpoint.keys()))
            model_keys = set(list(model_state_dict.keys()))
            common_keys = list(pretrained_keys & model_keys)
            checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
            # if checkpoint['pos_embed.proj.weight'].shape != model.pos_embed.proj.weight.shape and checkpoint['pos_embed.proj.weight'].ndim == 4:
            #     logger.info(f"Resize pos_embed, {checkpoint['pos_embed.proj.weight'].shape} -> {model.pos_embed.proj.weight.shape}")
            #     repeat = model.pos_embed.proj.weight.shape[2]
            #     checkpoint['pos_embed.proj.weight'] = checkpoint['pos_embed.proj.weight'].unsqueeze(2).repeat(1, 1, repeat, 1, 1) / float(repeat)
                # del checkpoint['proj_out.weight'], checkpoint['proj_out.bias']
        else:  # latest stage training weight
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        logger.info(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        logger.info(f'Successfully load {len(model_state_dict) - len(missing_keys)}/{len(model_state_dict)} keys from {args.pretrained}!')

    # Freeze main model
    ae.vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    model.requires_grad_(False)

    # load image encoder
    if args.image_encoder_type == 'clip':
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir="/storage/cache_dir")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_name, cache_dir=args.cache_dir)
    elif args.image_encoder_type == 'dino':
        # self.image_encoder = AutoModel.from_pretrained("facebook/dinov2-giant", cache_dir="/storage/cache_dir")
        image_encoder = AutoModel.from_pretrained(args.image_encoder_name, cache_dir=args.cache_dir)
    else:
        raise NotImplementedError
    
    image_encoder.requires_grad_(False)

    if args.pretrained_vip_adapter_path is None:
        attn_proc_type_dict = {}        
        for name, attn_processor in model.attn_processors.items():
            # replace all attn2.processor with VideoIPAttnProcessor
            if name.endswith('.attn2.processor'):
                attn_proc_type_dict[name] = 'VideoIPAttnProcessor'
            else:
                attn_proc_type_dict[name] = attn_processor.__class__.__name__

        vip = VIPNet(
            image_encoder_type=args.image_encoder_type,
            cross_attention_dim=1152,
            num_tokens=272, # NOTE should be modified 
            vip_inner_dim=1024,
            vip_num_attention_layers=2,
            attention_mode=args.attention_mode,
            gradient_checkpointing=args.gradient_checkpointing,
            vae_scale_factor_t=ae_stride_t,
            video_length=latent_size_t,
            attn_proc_type_dict=attn_proc_type_dict,
        )

    else:
        vip = VIPNet.from_pretrained(args.pretrained_vip_adapter_path)

    vip.train()

    init_from_original_attn_processor = True if args.pretrained_vip_adapter_path is None else False
    vip.set_vip_adapter(model, init_from_original_attn_processor=init_from_original_attn_processor)

    # for name, module in vip.named_modules():
    # #     # if isinstance(module, VideoIPAttnProcessor):
    # #     #     module.register_full_backward_hook(hook_backward_fn)
    #     if isinstance(module, nn.Conv3d):
    #         module.register_backward_hook(hook_backward_fn)

    noise_scheduler = DDPMScheduler(rescale_betas_zero_snr=args.zero_terminal_snr)
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    ae.vae.to(accelerator.device, dtype=torch.float32)
    # ae.vae.to(accelerator.device, dtype=weight_dtype)
    text_enc.to(accelerator.device, dtype=weight_dtype)
    model.to(accelerator.device, dtype=weight_dtype)

    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_vip = deepcopy(vip)
        ema_vip = EMAModel(vip.parameters(), update_after_step=args.ema_start_step,
                             model_cls=VIPNet, model_config=ema_vip.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_vip.save_pretrained(os.path.join(output_dir, "model_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "model"))
                    if weights:  # Don't pop if empty
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), VIPNet)
                ema_vip.load_state_dict(load_model.state_dict())
                ema_vip.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = VIPNet.from_pretrained(input_dir, subfolder="model")
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

    params_to_optimize = vip.parameters()
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

    # Setup data:
    train_dataset = getdataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # pin_memory=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        # prefetch_factor=8
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
    vip, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vip, optimizer, train_dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_vip.to(accelerator.device)

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

    logger.info("***** Running training *****")
    logger.info(f"  Model = {vip}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total trainable parameters = {sum(p.numel() for p in vip.parameters() if p.requires_grad) / 1e9} B")
    global_step = 0
    first_epoch = 0

    # NOTE checking update of params 
    # initial_params = {name: param.clone() for name, param in vip.named_parameters()}

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

    def sync_gradients_info(loss):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if args.use_ema:
            ema_vip.step(vip.parameters())
        progress_bar.update(1)
        progress_info.global_step += 1
        end_time = time.time()
        accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
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

        noise = torch.randn_like(model_input)
        if args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.noise_offset * torch.randn((model_input.shape[0], model_input.shape[1], 1, 1, 1),
                                                     device=model_input.device)

        bsz = model_input.shape[0]
        # Sample a random timestep for each image without bias.
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
       
        model_pred = vip(
            model=model,
            latent_model_input=noisy_model_input,
            timestep=timesteps,
            **model_kwargs,
        )

        model_pred = torch.chunk(model_pred, 2, dim=1)[0]
        

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

        if args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
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
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()


        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        progress_info.train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params_to_clip = vip.parameters()
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
                        args=args, 
                        model=model, 
                        vip=vip,
                        vae=ae, 
                        text_encoder=text_enc.text_enc,
                        tokenizer=train_dataset.tokenizer, 
                        image_processor=train_dataset.image_processor,
                        image_encoder=image_encoder,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype, 
                        global_step=progress_info.global_step,
                    )

                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_vip.store(vip.parameters())
                    ema_vip.copy_to(vip.parameters())
                    log_validation(
                        args=args, 
                        model=model, 
                        vip=vip,
                        vae=ae, 
                        text_encoder=text_enc.text_enc,
                        tokenizer=train_dataset.tokenizer, 
                        image_processor=train_dataset.image_processor,
                        image_encoder=image_encoder,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype, 
                        global_step=progress_info.global_step,
                        ema=True,
                    )
                    # Switch back to the original UNet parameters.
                    ema_vip.restore(vip.parameters())

        if prof is not None:
            prof.step()


        return loss

    def train_one_step(step_, data_item_, prof_=None):
        # NOTE checking params update
        # if step_ > 1:
        #     print("Comparing model parameters before and after training step:")
        #     for name, param in vip.named_parameters():
        #         if not torch.equal(param, initial_params[name]):
        #             print(f"Parameter {name} has changed.")
        #             initial_params[name] = param.clone()
        #         else:
        #             print(f"Parameter {name} has not changed!")
        
        train_loss = 0.0
        x, attn_mask, input_ids, cond_mask, clip_data = data_item_
        # Sample noise that we'll add to the latents

        if not args.multi_scale:
            assert torch.all(attn_mask)
        assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
        x = x.to(accelerator.device, dtype=ae.vae.dtype)  # B C T+image_num H W
        clip_data = clip_data.to(accelerator.device)  # B T+image_num C H W

        attn_mask = attn_mask.to(accelerator.device)  # B T+image_num H W
        input_ids = input_ids.to(accelerator.device)  # B 1+image_num L
        cond_mask = cond_mask.to(accelerator.device)  # B 1+image_num L
        # print('x.shape, attn_mask.shape, input_ids.shape, cond_mask.shape', x.shape, attn_mask.shape, input_ids.shape, cond_mask.shape)

        with torch.no_grad():
            # import ipdb;ipdb.set_trace()
            # use for loop to avoid OOM, because T5 is too huge...
            B, N, L = input_ids.shape  # B 1+image_num L
            # cond_ = torch.stack([text_enc(input_ids[i], cond_mask[i]) for i in range(B)])  # B 1+num_images L D

            # use batch inference
            input_ids_ = input_ids.reshape(-1, L)
            cond_mask_ = cond_mask.reshape(-1, L)
            cond = text_enc(input_ids_, cond_mask_)  

            assert not torch.any(torch.isnan(cond)), 'after text_enc'

            cond = cond.reshape(B, N, L, -1) # B 1+image_num L D

            clip_data = rearrange(clip_data, 'b t c h w -> (b t) c h w') # B T+image_num C H W -> B * (T+image_num) C H W
            clip_feature = image_encoder(clip_data).last_hidden_state # B * (T+image_num) N D
            clip_feature = clip_feature[:, 1:] # drop cls token
            clip_feature_height = clip_feature_width = int(sqrt(clip_feature.shape[1]))
            clip_feature = rearrange(clip_feature, '(b t) (h w) c -> b t h w c', b=B, h=clip_feature_height, w=clip_feature_width) # B T+image_num H W D  

            # Map input images to latent space + normalize latents
            if args.use_image_num == 0:
                x = ae.encode(x)  # B C T H W
            else:
                videos, images = x[:, :, :-args.use_image_num], x[:, :, -args.use_image_num:]
                videos = ae.encode(videos)  # B C T H W
                images = rearrange(images, 'b c t h w -> (b t) c 1 h w')
                images = ae.encode(images)
                images = rearrange(images, '(b t) c 1 h w -> b c t h w', t=args.use_image_num)
                x = torch.cat([videos, images], dim=2)  # b c 17+4, h, w

            assert not torch.any(torch.isnan(x)), 'after vae'
           
        with accelerator.accumulate(vip):
            x = x.to(weight_dtype)
            assert not torch.any(torch.isnan(x)), 'after vae' 
            model_kwargs = dict(encoder_hidden_states=cond, attention_mask=attn_mask,
                                encoder_attention_mask=cond_mask, use_image_num=args.use_image_num,
                                clip_feature=clip_feature)
            run(x, model_kwargs, prof_)

        if progress_info.global_step >= args.max_train_steps:
            return True

        return False

    def train_all_epoch(prof_=None):
        for epoch in range(first_epoch, args.num_train_epochs):
            progress_info.train_loss = 0.0
            if progress_info.global_step >= args.max_train_steps:
                return True

            for step, data_item in enumerate(train_dataloader):
                if accelerator.is_main_process:
                    if progress_info.global_step == 0:
                        print("before training, we need to check the validation mode...")
                        log_validation(
                            args=args, 
                            model=model, 
                            vip=vip,
                            vae=ae, 
                            text_encoder=text_enc.text_enc,
                            tokenizer=train_dataset.tokenizer, 
                            image_processor=train_dataset.image_processor,
                            image_encoder=image_encoder,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype, 
                            global_step=progress_info.global_step,
                        )
                if train_one_step(step, data_item, prof_):
                    break

    train_all_epoch()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--video_data", type=str, required='')
    parser.add_argument("--image_data", type=str, default='')
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
   

    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument('--cfg', type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")

    # text encoder & vae & diffusion model
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="Latte-XL/122")
    parser.add_argument('--enable_8bit_t5', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
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
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # diffusion setting
    parser.add_argument("--zero_terminal_snr", action="store_true", help="Whether to zero the terminal SNR.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--prediction_type", type=str, default=None, help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")

    # validation & logs
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
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

    # inpaint dataset
    parser.add_argument("--i2v_ratio", type=float, default=0.5) # for inpainting mode
    parser.add_argument("--transition_ratio", type=float, default=0.4) # for inpainting mode
    parser.add_argument("--default_text_ratio", type=float, default=0.1)
    parser.add_argument("--validation_dir", type=str, default=None, help="Path to the validation dataset.")
    parser.add_argument("--image_encoder_type", type=str, default='clip', choices=['clip', 'dino'])
    parser.add_argument("--image_encoder_name", type=str, default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    parser.add_argument("--pretrained_vip_adapter_path", type=str, default=None)
    parser.add_argument("--clear_video_ratio", type=float, default=0.0)
    parser.add_argument("--use_image_num", type=int, default=0)


    args = parser.parse_args()
    main(args)
