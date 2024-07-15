#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys
import gradio as gr
from diffusers import ConsistencyDecoderVAE, DPMSolverMultistepScheduler, Transformer2DModel, AutoencoderKL, SASolverScheduler

import torch
from typing import Tuple
from datetime import datetime
from peft import PeftModel
# import spaces
from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, MT5EncoderModel
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.serve.gradio_utils import DESCRIPTION, MAX_SEED, style_list, randomize_seed_fn, save_video

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
CACHE_EXAMPLES = False
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "6000"))
MAX_VIDEO_FRAME = int(os.getenv("MAX_IMAGE_SIZE", "93"))
SPEED_UP_T5 = os.getenv("USE_TORCH_COMPILE", "0") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
PORT = int(os.getenv("DEMO_PORT", "15432"))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(Default)"
SCHEDULE_NAME = [
    "PNDM-Solver", "EulerA-Solver", "DPM-Solver", "SA-Solver", 
    "DDIM-Solver", "Euler-Solver", "DDPM-Solver", "DEISM-Solver"]
DEFAULT_SCHEDULE_NAME = "PNDM-Solver"
NUM_IMAGES_PER_PROMPT = 1

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

if torch.cuda.is_available():
    weight_dtype = torch.bfloat16
    T5_token_max_length = 512

    vae = getae_wrapper('CausalVAEModel_4x8x8')("/storage/dataset/test140k")
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = 0.125
    vae.vae.tile_sample_min_size = 256
    vae.vae.tile_latent_min_size = 32
    vae.vae.tile_sample_min_size_t = 29
    vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config['CausalVAEModel_4x8x8']

    text_encoder = MT5EncoderModel.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", 
                                                   low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl")
    transformer = OpenSoraT2V.from_pretrained("/storage/dataset/hw29/model_ema", low_cpu_mem_usage=False, 
                                              device_map=None, torch_dtype=weight_dtype)
    scheduler = PNDMScheduler()
    pipe = OpenSoraPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler, 
                                transformer=transformer)
    pipe.to(device)
    print("Loaded on Device!")

    # speed-up T5
    if SPEED_UP_T5:
        pipe.text_encoder.to_bettertransformer()

    if USE_TORCH_COMPILE:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")

# @spaces.GPU(duration=120)
@torch.no_grad()
@torch.inference_mode()
def generate(
        prompt: str,
        negative_prompt: str = "",
        style: str = DEFAULT_STYLE_NAME,
        use_negative_prompt: bool = False,
        seed: int = 0,
        frame: int = 29,
        schedule: str = 'DPM-Solver',
        guidance_scale: float = 4.5,
        num_inference_steps: int = 25,
        randomize_seed: bool = False,
        progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    if schedule == 'DPM-Solver':
        if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            pipe.scheduler = DPMSolverMultistepScheduler()
    elif schedule == "PNDM-Solver":
        if not isinstance(pipe.scheduler, PNDMScheduler):
            pipe.scheduler = PNDMScheduler()
    elif schedule == "DDIM-Solver":
        if not isinstance(pipe.scheduler, DDIMScheduler):
            pipe.scheduler = DDIMScheduler()
    elif schedule == "Euler-Solver":
        if not isinstance(pipe.scheduler, EulerDiscreteScheduler):
            pipe.scheduler = EulerDiscreteScheduler()
    elif schedule == "DDPM-Solver":
        if not isinstance(pipe.scheduler, DDPMScheduler):
            pipe.scheduler = DDPMScheduler()
    elif schedule == "EulerA-Solver":
        if not isinstance(pipe.scheduler, EulerAncestralDiscreteScheduler):
            pipe.scheduler = EulerAncestralDiscreteScheduler()
    elif schedule == "DEISM-Solver":
        if not isinstance(pipe.scheduler, DEISMultistepScheduler):
            pipe.scheduler = DEISMultistepScheduler()
    elif schedule == "SA-Solver":
        if not isinstance(pipe.scheduler, SASolverScheduler):
            pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction', tau_func=lambda t: 1 if 200 <= t <= 800 else 0, predictor_order=2, corrector_order=2)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
    print(prompt, negative_prompt)
    videos = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=frame,
        # width=1280,
        # height=720,
        width=640,
        height=480,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        num_images_per_prompt=1,  # num_imgs
        max_sequence_length=T5_token_max_length,
    ).images

    video_paths = [save_video(vid) for vid in videos]
    print(video_paths)
    return video_paths[0], seed


examples = [
    "A small cactus with a happy face in the Sahara desert.",
    "Eiffel Tower was Made up of more than 2 million translucent straws to look like a cloud, with the bell tower at the top of the building, Michel installed huge foam-making machines in the forest to blow huge amounts of unpredictable wet clouds in the building's classic architecture.",
    "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest.",
    "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park.",
    "A close-up photo of a person. The subject is a woman. She wore a blue coat with a gray dress underneath. She has blue eyes and blond hair, and wears a pair of earrings. Behind are blurred city buildings and streets.",
    "A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.",
    "a handsome young boy in the middle with sky color background wearing eye glasses, it's super detailed with anime style, it's a portrait with delicated eyes and nice looking face",
    "an astronaut sitting in a diner, eating fries, cinematic, analog film",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
    "The parametric hotel lobby is a sleek and modern space with plenty of natural light. The lobby is spacious and open with a variety of seating options. The front desk is a sleek white counter with a parametric design. The walls are a light blue color with parametric patterns. The floor is a light wood color with a parametric design. There are plenty of plants and flowers throughout the space. The overall effect is a calm and relaxing space. occlusion, moody, sunset, concept art, octane rendering, 8k, highly detailed, concept art, highly detailed, beautiful scenery, cinematic, beautiful light, hyperreal, octane render, hdr, long exposure, 8K, realistic, fog, moody, fire and explosions, smoke, 50mm f2.8",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Row(equal_height=False):
        with gr.Group():
            with gr.Row():
                use_negative_prompt = gr.Checkbox(label="Use additional negative prompt", value=False, visible=True)
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a additional negative prompt",
                visible=True,
            )
            with gr.Row(visible=True):
                schedule = gr.Radio(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=SCHEDULE_NAME,
                    value=DEFAULT_SCHEDULE_NAME,
                    label="Sampler Schedule",
                    visible=True,
                )
            style_selection = gr.Radio(
                show_label=True,
                container=True,
                interactive=True,
                choices=STYLE_NAMES,
                value=DEFAULT_STYLE_NAME,
                label="Video Style",
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row(visible=True):
                frame = gr.Slider(
                    label="Frame",
                    minimum=29,
                    maximum=MAX_VIDEO_FRAME,
                    step=16,
                    value=29,
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=5.0,
                )
                inference_steps = gr.Slider(
                    label="inference steps",
                    minimum=10,
                    maximum=200,
                    step=1,
                    value=50,
                )
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Video(label="Result")

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            seed,
            frame,
            schedule,
            guidance_scale,
            inference_steps,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    # demo.queue(max_size=20).launch(server_name='0.0.0.0', share=True)
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=11900, debug=True)