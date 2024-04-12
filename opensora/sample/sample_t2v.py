import math
import os
import argparse
import os, sys
from typing import List, Union

import imageio
import torch
from torchvision.utils import save_image
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from transformers import T5EncoderModel, T5Tokenizer

from opensora.models.ae import ae_stride_config, getae_wrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.utils.utils import save_video_grid
sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen import VideoGenPipeline


def get_models(args: argparse.Namespace, device: str):
    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", ).to(device, dtype=torch.float16)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    # Load model:
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, torch_dtype=torch.float16).to(device)
    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, )
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()
    
    return transformer_model, vae, text_encoder, tokenizer


def get_scheduler(sample_method: str):
    schedulers = {
        'DDIM': DDIMScheduler(),
        'EulerDiscrete': EulerDiscreteScheduler(),
        'DDPM': DDPMScheduler(),
        'DPMSolverMultistep': DPMSolverMultistepScheduler(),
        'DPMSolverSinglestep': DPMSolverSinglestepScheduler(),
        'PNDM': PNDMScheduler(),
        'HeunDiscrete': HeunDiscreteScheduler(),
        'EulerAncestralDiscrete': EulerAncestralDiscreteScheduler(),
        'DEISMultistep': DEISMultistepScheduler(),
        'KDPM2AncestralDiscrete': KDPM2AncestralDiscreteScheduler()
    }
    return schedulers[sample_method]


def get_text_prompt(text_prompt: Union[List[str], str]):
    if not isinstance(text_prompt, list):
        text_prompt = [text_prompt]
        
    if len(text_prompt) == 1 and text_prompt[0].endswith('txt'):
        text_prompt = open(text_prompt[0], 'r').readlines()
        text_prompt = [i.strip() for i in text_prompt]
    return text_prompt


def save_video(videos: torch.FloatTensor, prompt: str, args: argparse.Namespace):
    """
    Save a single video (output of pipeline).
    """
    # Save results
    try:
        if args.force_images:
            videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
            save_image(
                videos / 255.0, 
                os.path.join(
                    args.save_img_path,
                    prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{args.ext}'),
                nrow=1, normalize=True, value_range=(0, 1))  # t c h w

        else:
            imageio.mimwrite(
                os.path.join(
                    args.save_img_path,
                    prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{args.ext}'
                ), 
                videos[0],
                fps=args.fps, 
                quality=9)  # highest quality is 10, lowest is 0
    except:
        print('Error when saving {}'.format(prompt))
    
    return videos


def save_grid(video_grids: List[torch.FloatTensor], args: argparse.Namespace):
    video_grids = torch.cat(video_grids, dim=0)
        
    # Save results
    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    if args.force_images:
        save_image(
            video_grids / 255.0, 
            os.path.join(
                args.save_img_path, 
                f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{args.ext}'),
            nrow=math.ceil(math.sqrt(len(video_grids))), 
            normalize=True, value_range=(0, 1))
    else:
        video_grids = save_video_grid(video_grids)
        imageio.mimwrite(
            os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{args.ext}'), 
            video_grids, 
            fps=args.fps, 
            quality=9)

    print('save path {}'.format(args.save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")
    
    
def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('videogen_pipeline', device)
    
    
    # Prepare models and pipeline
    transformer_model, vae, text_encoder, tokenizer = get_models(args, device)
    scheduler = get_scheduler(args.sample_method)
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model,
                                         )
    # Some pipeline configs
    # videogen_pipeline.enable_sequential_cpu_offload()
    # videogen_pipeline.enable_xformers_memory_efficient_attention()


    # Prepare
    video_length, image_size = transformer_model.config.video_length, int(args.version.split('x')[1])
    latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    vae.latent_size = latent_size
    if args.force_images:
        video_length = 1
        args.ext = 'jpg'
    else:
        args.ext = 'mp4'

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)


    # Get text prompts
    text_prompt = get_text_prompt(args.text_prompt)


    # Video generation
    video_grids = []
    for prompt in text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt,
                                   video_length=video_length,
                                   height=image_size,
                                   width=image_size,
                                   num_inference_steps=args.num_sampling_steps,
                                   guidance_scale=args.guidance_scale,
                                   enable_temporal_attentions=not args.force_images,
                                   num_images_per_prompt=1,
                                   mask_feature=True,
                                   ).video
        
        videos = save_video(videos, prompt, args)
        
        # Save result
        if args.save_grid:
            video_grids.append(videos)
    
    
    # Save results
    if args.save_grid:
        save_grid(video_grids, args)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default='65x512x512', choices=['65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--save_grid', action='store_true', help='Save all prompts in a grid')
    args = parser.parse_args()

    main(args)