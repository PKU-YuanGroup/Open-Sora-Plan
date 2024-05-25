import os
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen import VideoGenPipeline

import json
import pandas as pd
import numpy as np
import imageio

def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/remote-home1/ysh/MagicBench/1_comparison/Open-Sora-NUS_batch/ckpts"

    vae = getae(args).to(device, dtype=torch.float16)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    if getae_wrapper(args) == CausalVQVAEModelWrapper or getae_wrapper(args) == CausalVAEModelWrapper:
        video_length = args.num_frames // ae_stride_config[args.ae][0] + 1
    else:
        video_length = args.num_frames // ae_stride_config[args.ae][0]

    # Load model:
    latent_size = (args.image_size // ae_stride_config[args.ae][1], args.image_size // ae_stride_config[args.ae][2])
    args.latent_size = latent_size
    vae.latent_size = latent_size
    transformer_model = LatteT2V.from_pretrained(args.ckpt, subfolder="model", torch_dtype=torch.float16).to(device)

    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=cache_dir, torch_dtype=torch.float16).to(device)

    transformer_model.eval()  # important!

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':
        scheduler = DDPMScheduler()
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler()
    print('videogen_pipeline', device)
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model).to(device=device)
    if args.temporal_lora_path:
        from swift import Swift
        print("load lora from swift for Latte")
        Swift.from_pretrained(videogen_pipeline.transformer, args.temporal_lora_path)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    video_grids = []
    if args.prompts_file:
        with open(args.prompts_file, 'r') as file:
            prompts = file.read().splitlines()
    else:
        if not isinstance(args.text_prompt, list):
            args.text_prompt = [args.text_prompt]
        prompts = args.text_prompt if args.text_prompt else []

    if args.human:
            sample_idx = 0  # Initialize sample index
            while True:
                prompt = input("Enter your prompt (or type 'exit' to quit): ")
                if prompt.lower() == "exit":
                    break

                args.seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate a random seed if not provided
                torch.manual_seed(args.seed)  # Use the seed for torch operations
                print(f"current seed: {args.seed}")
                print('Processing the ({}) prompt'.format(prompt))
                videos = videogen_pipeline(prompt,
                                            video_length=video_length,
                                            height=args.image_size,
                                            width=args.image_size,
                                            num_inference_steps=args.num_sampling_steps,
                                            guidance_scale=args.guidance_scale,
                                            enable_temporal_attentions=True,
                                            num_images_per_prompt=1,
                                            mask_feature=True,
                                            ).video
                try:
                    imageio.mimwrite(
                            os.path.join(
                                args.save_img_path,
                                f"{idx}_seed{args.seed}_{prompt.replace(' ', '_')[:40]}_{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4"
                            ), videos[0],
                            fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
                except:
                    print('Error when saving {}'.format(prompt))
                    video_grids.append(videos)    
                sample_idx += 1
    elif args.run_csv or args.run_json:
        batch_size = args.batch_size

        assert args.run_csv != args.run_json

        if args.run_csv:
            print("run csv")
            output_video_path = f"./MoT-evaluation/"
            file_path = "/remote-home1/ysh/MagicBench/1_comparison/0_csv_and_json/chunk_csv/split_data_8.csv"
            data = pd.read_csv(file_path)
            prompts = data['name'].tolist()
            videoids = data['videoid'].tolist()
        else:
            print("run json")
            output_video_path = f"./MSRVTT/"
            file_path = "/remote-home1/ysh/MagicBench/1_comparison/0_csv_and_json/chunk_json/splitted_file_1.json"
            with open(file_path, 'r') as file:
                data = json.load(file)
            prompts = []
            videoids = []
            senids = []
            for item in data:
                prompts.append(item['caption'])
                videoids.append(item['video_id'])
                senids.append(item['sen_id'])

        os.makedirs(output_video_path, exist_ok=True)
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts_raw = prompts[i : i + batch_size]
            batch_prompts = [prompt for prompt in batch_prompts_raw]

            if args.run_csv or args.run_json:
                batch_videoids = videoids[i : i + batch_size]
            if args.run_json:
                batch_senids = senids[i : i + batch_size]

            args.seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate a random seed if not provided
            torch.manual_seed(args.seed)  # Use the seed for torch operations
            print(f"current seed: {args.seed}")
            
            videos = videogen_pipeline(batch_prompts,
                                        video_length=video_length,
                                        height=args.image_size,
                                        width=args.image_size,
                                        num_inference_steps=args.num_sampling_steps,
                                        guidance_scale=args.guidance_scale,
                                        enable_temporal_attentions=True,
                                        num_images_per_prompt=1,
                                        mask_feature=True,
                                    ).video
            
            for idx, sample in enumerate(videos):
                if args.run_csv:
                    new_filename = f"{batch_videoids[idx]}.mp4"
                if args.run_json:
                    new_filename = f"{batch_videoids[idx]}-{batch_senids[idx]}.mp4"
                
                imageio.mimwrite(os.path.join(output_video_path, new_filename), sample, fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
    else:
        for idx, prompt in enumerate(prompts):
            args.seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate a random seed if not provided
            torch.manual_seed(args.seed)  # Use the seed for torch operations
            print(f"current seed: {args.seed}")
            print('Processing the ({}) prompt'.format(prompt))
            videos = videogen_pipeline(prompt,
                                    video_length=video_length,
                                    height=args.image_size,
                                    width=args.image_size,
                                    num_inference_steps=args.num_sampling_steps,
                                    guidance_scale=args.guidance_scale,
                                    enable_temporal_attentions=True,
                                    num_images_per_prompt=1,
                                    mask_feature=True,
                                    ).video
            try:
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f"{idx}_seed{args.seed}_{prompt.replace(' ', '_')[:40]}_{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4"
                    ), videos[0],
                    fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
            except:
                print('Error when saving {}'.format(prompt))
            video_grids.append(videos)

    video_grids = torch.cat(video_grids, dim=0)

    video_grids = save_video_grid(video_grids)

    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    imageio.mimwrite(os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.mp4'), video_grids, fps=args.fps, quality=9)
    print('save path {}'.format(args.save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model", type=str, default='Latte-XL/122')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--variance_type", type=str, default="learned_range")
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')

    parser.add_argument("--temporal_lora_path", type=str, default=None)

    parser.add_argument("--prompts_file", type=str, help="Path to the text file containing prompts, one per line.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--human", action="store_true", help="Enable human mode for interactive video generation")
    parser.add_argument("--run_csv", type=str, default=None)
    parser.add_argument("--run_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    args = parser.parse_args()

    main(args)