import gradio as gr
import os
import torch
from einops import rearrange
import torch.distributed as dist
from torchvision.utils import save_image
import imageio
import math
import argparse
import random
import numpy as np
import string

from opensora.sample.caption_refiner import OpenSoraCaptionRefiner
from opensora.utils.sample_utils import (
    prepare_pipeline, save_video_grid, init_gpu_env
)
from .gradio_utils import *



@torch.no_grad()
@torch.inference_mode()
def generate(
        prompt: str,
        image_1: str, 
        image_2: str = None, 
        seed: int = 0,
        num_frames: int = 29, 
        num_samples: int = 1, 
        guidance_scale: float = 4.5,
        num_inference_steps: int = 25,
        randomize_seed: bool = False,
        progress=gr.Progress(track_tqdm=True),
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    if seed is not None:
        torch.manual_seed(seed)
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    text_prompt = [prompt]
    images = [[image_1] if image_2 is None else [image_1, image_2]]
    

    for index, (image, prompt) in enumerate(zip(images, text_prompt)):
        if caption_refiner_model is not None:
            refine_prompt = caption_refiner_model.get_refiner_output(prompt)
            print(f'\nOrigin prompt: {prompt}\n->\nRefine prompt: {refine_prompt}')
            prompt = refine_prompt
        input_prompt = POS_PROMPT.format(prompt)
        print(image)
        videos = pipeline(
            conditional_images=image, 
            prompt=input_prompt, 
            negative_prompt=NEG_PROMPT, 
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_samples_per_prompt=num_samples,
            max_sequence_length=512,
            device=device, 
            ).videos
        if num_frames != 1 and enhance_video_model is not None:
            # b t h w c
            videos = enhance_video_model.enhance_a_video(videos, input_prompt, 2.0, args.fps, 250)
        if num_frames == 1:
            videos = rearrange(videos, 'b t h w c -> (b t) c h w')
            if num_samples != 1:
                for i, image in enumerate(videos):
                    save_image(
                        image / 255.0, 
                        os.path.join(
                            args.save_img_path, 
                            f'{args.sample_method}_{index}_gs{guidance_scale}_s{num_inference_steps}_i{i}.jpg'
                            ),
                        nrow=math.ceil(math.sqrt(videos.shape[0])), 
                        normalize=True, 
                        value_range=(0, 1)
                        )  # b c h w
            save_image(
                videos / 255.0, 
                os.path.join(
                    args.save_img_path, 
                    f'{args.sample_method}_{index}_gs{guidance_scale}_s{num_inference_steps}.jpg'
                    ),
                nrow=math.ceil(math.sqrt(videos.shape[0])), 
                normalize=True, 
                value_range=(0, 1)
                )  # b c h w
        else:
            if num_samples == 1:
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f'{args.sample_method}_{index}_gs{guidance_scale}_s{num_inference_steps}.mp4'
                    ), 
                    videos[0],
                    fps=args.fps, 
                    quality=6
                    )  # highest quality is 10, lowest is 0
            else:
                for i in range(num_samples):
                    imageio.mimwrite(
                        os.path.join(
                            args.save_img_path,
                            f'{args.sample_method}_{index}_gs{guidance_scale}_s{num_inference_steps}_i{i}.mp4'
                        ), videos[i],
                        fps=args.fps, 
                        quality=6
                        )  # highest quality is 10, lowest is 0
                    
                videos = save_video_grid(videos)
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        f'{args.sample_method}_{index}_gs{guidance_scale}_s{num_inference_steps}.mp4'
                    ), 
                    videos,
                    fps=args.fps, 
                    quality=6
                    )  # highest quality is 10, lowest is 0)
                videos = videos.unsqueeze(0) # 1 t h w c
        video_grids.append(videos)

    video_grids = torch.cat(video_grids, dim=0)
    
    final_path = os.path.join(
                    args.save_img_path,
                    f'{args.sample_method}_gs{guidance_scale}_s{num_inference_steps}'
                    )

    random_string = ''.join(random.choices(string.ascii_letters, k=4))
    if num_frames == 1:
        final_path = final_path + f'_{random_string}.jpg'
        save_image(
            video_grids / 255.0, 
            final_path, 
            nrow=math.ceil(math.sqrt(len(video_grids))), 
            normalize=True, 
            value_range=(0, 1)
            )
    else:
        video_grids = save_video_grid(video_grids)
        final_path = final_path + f'_{random_string}.mp4'
        imageio.mimwrite(
            final_path, 
            video_grids, 
            fps=args.fps, 
            quality=6
            )
    print('save path {}'.format(args.save_img_path))
    return final_path, seed



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
parser.add_argument("--version", type=str, default='v1_3', choices=['v1_3', 'v1_5'])
parser.add_argument("--caption_refiner", type=str, default=None)
parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
parser.add_argument("--text_encoder_name_1", type=str, default='DeepFloyd/t5-v1_1-xxl')
parser.add_argument("--text_encoder_name_2", type=str, default=None)
parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
parser.add_argument("--fps", type=int, default=24)
parser.add_argument('--enable_tiling', action='store_true')
parser.add_argument('--save_memory', action='store_true')
parser.add_argument('--compile', action='store_true') 
parser.add_argument("--gradio_port", type=int, default=11900)
parser.add_argument("--enhance_video", type=str, default=None)
parser.add_argument("--model_type", type=str, default='i2v')
args = parser.parse_args()

args.model_path = "/storage/gyy/hw/Open-Sora-Plan/runs/inpaint_93x1280x1280_stage3_gpu/checkpoint-1692/model_ema"
args.version = "v1_3"
args.caption_refiner = "/storage/ongoing/refine_model/llama3_1_instruct_lora/llama3_8B_lora_merged_cn"
args.ae = "WFVAEModel_D8_4x8x8"
args.ae_path = "/storage/lcm/wf-vae_trilinear"
args.text_encoder_name_1 = "/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl"
args.text_encoder_name_2 = None
args.save_img_path = "./test_gradio"
args.fps = 18

args.prediction_type = "v_prediction"
args.rescale_betas_zero_snr = True
args.cache_dir = "./cache_dir"
args.sample_method = 'EulerAncestralDiscrete'
args.sp = False
args.crop_for_hw = False
args.max_hw_square = 1048576
args.enable_tiling = True

dtype = torch.bfloat16
args = init_gpu_env(args)
device = torch.cuda.current_device()

if args.enhance_video is not None:
    from opensora.sample.VEnhancer.enhance_a_video import VEnhancer
    enhance_video_model = VEnhancer(model_path=args.enhance_video, version='v2', device=device)
else:
    enhance_video_model = None

pipeline = prepare_pipeline(args, dtype, device)
if args.caption_refiner is not None:
    caption_refiner_model = OpenSoraCaptionRefiner(args, dtype, device)
else:
    caption_refiner_model = None

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(LOGO)
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)

    with gr.Row(equal_height=False):
        with gr.Group():
            with gr.Row():
                image_1 = gr.Image(type="filepath", label='Image 1')
                image_2 = gr.Image(type="filepath", label='Image 2')
            with gr.Row():
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                num_frames = gr.Slider(
                        label="Num Frames",
                        minimum=29,
                        maximum=93,
                        step=16,
                        value=29,
                    )
                num_samples = gr.Slider(
                        label="Num Samples",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                    )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    value=7.5,
                )
                inference_steps = gr.Slider(
                    label="Inference steps",
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
            result = gr.Video(autoplay=True, label="Result")
            # result = gr.Gallery(label="Result", columns=NUM_IMAGES_PER_PROMPT,  show_label=False)


    

    # with gr.Row(), gr.Column():
    #     gr.Markdown("## Examples (Text-to-Video)")
    #     examples = [[i, 42, 93, 1, 7.5, 100, False] for i in t2v_prompt_examples]
    #     gr.Examples(
    #         examples=examples, 
    #         inputs=[
    #             prompt, seed, num_frames, num_samples, 
    #             guidance_scale, inference_steps, randomize_seed
    #             ],
    #         label='Text-to-Video', 
    #         cache_examples=False, 
    #         outputs=[result, seed],
    #         fn=generate
    #         )


    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            image_1, 
            image_2, 
            seed,
            num_frames, 
            num_samples, 
            guidance_scale,
            inference_steps,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )



# if __name__ == "__main__":
demo.queue(max_size=20).launch(
    server_name="0.0.0.0", 
    server_port=args.gradio_port+args.local_rank, 
    debug=True
    )