

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
from opensora.serve.gradio_utils import block_css, title_markdown, randomize_seed_fn, set_env, examples


@torch.inference_mode()
def generate_img(prompt, sample_steps, scale, seed=0, randomize_seed=False, force_images=False):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)

    videos = videogen_pipeline(prompt,
                               video_length=transformer_model.config.video_length,
                               height=int(args.version.split('x')[1]),
                               width=int(args.version.split('x')[2]),
                               num_inference_steps=sample_steps,
                               guidance_scale=scale,
                               enable_temporal_attentions=not force_images,
                               num_images_per_prompt=1,
                               mask_feature=True,
                               ).video

    torch.cuda.empty_cache()
    videos = videos[0]
    tmp_save_path = 'tmp.mp4'
    imageio.mimwrite(tmp_save_path, videos, fps=24, quality=9)  # highest quality is 10, lowest is 0
    display_model_info = f"Image size: {int(args.version.split('x')[1])}, \nSampling Step: {sample_steps}, \nGuidance Scale: {scale}"
    return tmp_save_path, prompt, display_model_info, seed

if __name__ == '__main__':
    args = type('args', (), {
        'force_images': False,
        'model_path': 'LanguageBind/Open-Sora-Plan-65x512x512-v1.0.0',
        'text_encoder_name': 'DeepFloyd/t5-v1_1-xxl',
        'version': '65x512x512'
    })
    device = torch.device('cuda:0')
    # Load model:
    transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, torch_dtype=torch.float16, cache_dir='cache_dir').to(device)

    vae = CausalVAEModelWrapper(args.model_path, subfolder="vae", cache_dir='cache_dir').to(device, dtype=torch.float16)
    vae.vae.enable_tiling()

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
                                         transformer=transformer_model).to(device=device)

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
                ],
        outputs=[Video(label="Vid", width=512, height=512),
                 Textbox(label="input prompt"),
                 Textbox(label="model info"),
                 gr.Slider(label='seed')],
        title=title_markdown, theme=gr.themes.Default(), css=block_css,
        examples=examples,
    )
    demo.launch()