import math
import os
from tkinter.filedialog import Open
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder, Transformer2DModel
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.diffusion.udit.modeling_udit import UDiTT2V

from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.sample.pipeline_inpaint import OpenSoraInpaintPipeline

from opensora.models.diffusion.latte.modeling_inpaint import OpenSoraInpaint
from diffusers.configuration_utils import register_to_config 

import imageio

from PIL import Image
from einops import rearrange
import glob
from torchvision import transforms
from torchvision.transforms import Lambda
from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop, LongSideResizeVideo, SpatialStrideCropVideo
import numpy as np

def save_video(video, save_path='output_video.mp4', fps=24):
    import cv2

    frame_count, height, width, channels = video.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for i in range(frame_count):
        frame = video[i].cpu().numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        out.write(frame)

def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    weight_dtype = torch.float16
    device = torch.device(args.device)

    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir)
    vae = getae_wrapper(args.ae)(args.ae_path)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.vae_scale_factor = ae_stride_config[args.ae]

    # if args.model_3d:
    #     transformer_model = OpenSoraT2V.from_pretrained(args.model_path, subfolder=args.version, cache_dir=args.cache_dir, low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
    # else:
    #     transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, cache_dir=args.cache_dir, low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
    
    if args.model_3d:
        # transformer_model = OpenSoraT2V.from_pretrained(args.model_path, cache_dir=args.cache_dir, 
        #                                                 low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
        transformer_model = UDiTT2V.from_pretrained(args.model_path, cache_dir=args.cache_dir, 
                                                        low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
    else:
        transformer_model = OpenSoraInpaint(
            num_layers=28, 
            attention_head_dim=72, 
            num_attention_heads=16, 
            patch_size_t=1, 
            patch_size=2,
            norm_type="ada_norm_single", 
            caption_channels=4096, 
            cross_attention_dim=1152, 
            in_channels=4,
            out_channels=8, # 因为要加载预训练权重，所以这里out_channels仍然设置为2倍
            # caption_channels=4096,
            # cross_attention_dim=1152,
            attention_bias=True,
            sample_size=(64, 64),
            sample_size_t=17,
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
            attention_mode='xformers',
            downsampler=None,
            compress_kv_factor=1,
            use_rope=False,
            model_max_length=300,
        )
        transformer_model.custom_load_state_dict(args.model_path)
    # text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    # tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    
    
    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    vae.vae = vae.vae.to(device=device, dtype=torch.float32)
    transformer_model = transformer_model.to(device=device, dtype=weight_dtype)
    text_encoder = text_encoder.to(device=device, dtype=weight_dtype)

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler(clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler(clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif args.sample_method == 'EulerDiscreteSVD':
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid", 
                                                        subfolder="scheduler", cache_dir=args.cache_dir)

    print(args.sample_method, scheduler.__class__.__name__)

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

    resize = [CenterCropResizeVideo((args.height, args.width)),]
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])

    pipeline = OpenSoraInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer_model
    )

    pipeline.to(device)

    def preprocess_images(images):
        if len(images) == 1:
            condition_images_indices = [0]
        elif len(images) == 2:
            condition_images_indices = [0, -1]
        condition_images = [Image.open(image).convert("RGB") for image in images]
        condition_images = [torch.from_numpy(np.copy(np.array(image))) for image in condition_images]
        condition_images = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in condition_images]
        condition_images = [transform(image).to(device=device, dtype=torch.float32) for image in condition_images]
        return dict(condition_images=condition_images, condition_images_indices=condition_images_indices)
    
    videos = []
    for prompt, images in zip(validation_prompt, validation_images_list):
        if not isinstance(images, list):
            images = [images]
        print('Processing the ({}) prompt and the images ({})'.format(prompt, images))
        
        pre_results = preprocess_images(images)
        condition_images = pre_results['condition_images']
        condition_images_indices = pre_results['condition_images_indices']

        video = pipeline(
            prompt=prompt,
            condition_images=condition_images,
            condition_images_indices=condition_images_indices,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=True,
            num_images_per_prompt=1,
            mask_feature=True,
        ).images
        videos.append(video[0])
    
    # Save the generated videos
    save_dir = os.path.join(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    for idx, video in enumerate(videos):
        save_video(video, os.path.join(save_dir, f"video_{idx:06d}.mp4"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--model_3d', action='store_true')
    
    parser.add_argument("--validation_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)