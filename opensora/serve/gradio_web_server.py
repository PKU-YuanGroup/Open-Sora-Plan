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
# import spaces
from opensora.models import CausalVAEModelWrapper
from opensora.models.causalvideovae import ae_stride_config, ae_channel_config
from opensora.models.causalvideovae import ae_norm, ae_denorm
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
DEFAULT_SCHEDULE_NAME = "EulerA-Solver"
NUM_IMAGES_PER_PROMPT = 1




parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--ae_path", type=str, required=True)
parser.add_argument("--ae", type=str, default="CausalVAEModel_D4_4x8x8")
parser.add_argument("--text_encoder_name", type=str, default="google/mt5-xxl")
parser.add_argument("--cache_dir", type=str, default='./cache_dir')
args = parser.parse_args()

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

if torch.cuda.is_available():
    weight_dtype = torch.bfloat16
    T5_token_max_length = 512

    vae = CausalVAEModelWrapper(args.ae_path, args.cache_dir).eval()
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    vae.vae.enable_tiling()
    vae.vae.tile_overlap_factor = 0.125
    vae.vae.tile_sample_min_size = 256
    vae.vae.tile_latent_min_size = 32
    vae.vae.tile_sample_min_size_t = 29
    vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]

    # text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, 
    #                                                low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = MT5EncoderModel.from_pretrained("google/mt5-xxl", cache_dir=args.cache_dir, 
                                                   low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl", cache_dir=args.cache_dir)
    transformer = OpenSoraT2V.from_pretrained(args.model_path, cache_dir=args.cache_dir, low_cpu_mem_usage=False, 
                                              device_map=None, torch_dtype=weight_dtype)
    scheduler = EulerAncestralDiscreteScheduler()
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
        # frame: int = 29,
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
        num_frames=93,
        width=1280,
        height=720,
        # width=640,
        # height=480,
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
    "A young man at his 20s is sitting on a piece of cloud in the sky, reading a book.", 
    "an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt, he wears a brown beret and glasses and has a very professorial appearance, and the end he offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and city in the background, depth of field, cinematic 35mm film.", 
    "a woman’s face, illuminated by the soft light of dawn, her expression serene and content as she wakes up in a cozy bedroom.", 
    "a detective’s face, lit by a single desk lamp, his eyes scanning a wall covered in photos and notes, deep in thought.", 
    "Audience members in a theater are captured in a series of medium shots, with a young man and woman in formal attire centrally positioned and illuminated by a spotlight effect.", 
    "A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures.", 
    "a realistic 3d rendering of a female character with curly blonde hair and blue eyes. she is wearing a black tank top and has a neutral expression while facing the camera directly. the background is a plain blue sky, and the scene is devoid of any other objects or text. the character is detailed, with realistic textures and lighting, suitable for a video game or high-quality animation. there is no movement or additional action in the video. the focus is entirely on the character's appearance and realistic rendering.", 
    "A panda strumming a guitar under a bamboo grove, its paws gently plucking the strings as a group of mesmerized rabbits watch, the music blending with the rustle of bamboo leaves. HD.", 
    "a woman with a vintage hairstyle and bright red lipstick, gazing seductively into the camera, the background blurred to keep the focus solely on her.", 
    "In the jungle, a hidden temple stands guarded by statues of lions, their eyes glowing with emerald light, protecting secrets untold for millennia. 8K.", 
    "an old man’s weathered face, with deep wrinkles and a thick white mustache, looking out to sea, the wind gently blowing through his hair.", 
    "a soldier’s face, covered in dirt and sweat, his eyes filled with determination as he surveys the battlefield.", 
    "A river that flows uphill, defying gravity as it returns lost treasures from the sea to the mountain top, each item telling a story of a voyage gone by. HD.", 
    "a man’s face, lit only by the glow of his computer screen, his eyes wide and unblinking as he discovers something shocking online.", 
    "On a deserted island, palm trees sway to summon a rainstorm, their leaves conducting the wind like maestros, orchestrating a symphony of thunder and lightning. High Resolution.", 
    "a middle-aged man’s face, with a five o’clock shadow, staring pensively into the distance as rain softly taps against the window beside him, his thoughts deep and contemplative.", 
    "a man’s face, his expression one of deep concentration as he works on a complex task.", 
    "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach.The crashing blue waters create white-tipped waves,while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green", 
    "shrubbery covers the cliffs edge. The steep drop from the road down to the beach is adramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.", 
    "a woman standing in a dimly lit room. she is wearing a traditional chinese outfit, which includes a red and gold dress with intricate designs and a matching headpiece. the woman has her hair styled in an updo, adorned with a gold accessory. her makeup is done in a way that accentuates her features, with red lipstick and dark eyeshadow. she is looking directly at the camera with a neutral expression. the room has a rustic feel, with wooden beams and a stone wall visible in the background. the lighting in the room is soft and warm, creating a contrast with the woman's vibrant attire. there are no texts or other objects in the video. the style of the video is a portrait, focusing on the woman and her attire.",
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
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1,
                    maximum=20,
                    step=0.1,
                    value=10.0,
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
            # frame,
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