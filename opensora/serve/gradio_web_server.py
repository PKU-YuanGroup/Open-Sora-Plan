

import argparse
import sys
import os
import random

import imageio
import torch
from diffusers import PNDMScheduler
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from datetime import datetime
from typing import List, Union
import gradio as gr
import numpy as np
from gradio.components import Textbox, Video, Image
from transformers import T5Tokenizer, T5EncoderModel

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.sample.pipeline_videogen import VideoGenPipeline
from opensora.serve.gradio_utils import block_css, title_markdown, randomize_seed_fn, set_env, examples, DESCRIPTION


@torch.inference_mode()
def generate_img(prompt, sample_steps, scale, seed=0, randomize_seed=False, force_images=False):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)
    video_length = transformer_model.config.video_length if not force_images else 1
    height, width = int(args.version.split('x')[1]), int(args.version.split('x')[2])
    num_frames = 1 if video_length == 1 else int(args.version.split('x')[0])
    videos = videogen_pipeline(prompt,
                               num_frames=num_frames,
                               height=height,
                               width=width,
                               num_inference_steps=sample_steps,
                               guidance_scale=scale,
                               enable_temporal_attentions=not force_images,
                               num_images_per_prompt=1,
                               mask_feature=True,
                               ).video

    torch.cuda.empty_cache()
    videos = videos[0]
    tmp_save_path = 'tmp.mp4'
    imageio.mimwrite(tmp_save_path, videos, fps=24, quality=6)  # highest quality is 10, lowest is 0
    display_model_info = f"Video size: {num_frames}×{height}×{width}, \nSampling Step: {sample_steps}, \nGuidance Scale: {scale}"
    return tmp_save_path, prompt, display_model_info, seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.1.0')
    parser.add_argument("--version", type=str, default='65x512x512', choices=['65x512x512', '221x512x512'])
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument('--force_images', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:0')

    # Load model:
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, torch_dtype=torch.float16, cache_dir='cache_dir').to(device)
    
    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir').to(device)
    vae = vae.half()
    vae.vae.enable_tiling()
    vae.vae_scale_factor = ae_stride_config[args.ae]
    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir="cache_dir")
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir="cache_dir",
                                                  torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()
    scheduler = PNDMScheduler()
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model).to(device)


    demo = gr.Interface(
        fn=generate_img,
        inputs=[Textbox(label="",
                        placeholder="Please enter your prompt. \n"),
                gr.Slider(
                    label='Sample Steps',
                    minimum=1,
                    maximum=500,
                    value=50,
                    step=10
                ),
                gr.Slider(
                    label='Guidance Scale',
                    minimum=0.1,
                    maximum=30.0,
                    value=10.0,
                    step=0.1
                ),
                gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=203279,
                    step=1,
                    value=0,
                ),
                gr.Checkbox(label="Randomize seed", value=True),
                gr.Checkbox(label="Generate image (1 frame video)", value=False),
                ],
        outputs=[Video(label="Vid", width=512, height=512),
                 Textbox(label="input prompt"),
                 Textbox(label="model info"),
                 gr.Slider(label='seed')],
        title=title_markdown, description=DESCRIPTION, theme=gr.themes.Default(), css=block_css,
        examples=examples,
    )
    demo.launch()
