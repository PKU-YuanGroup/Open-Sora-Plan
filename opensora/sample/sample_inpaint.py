import math
import os
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
from transformers import T5EncoderModel, MT5EncoderModel, UMT5EncoderModel, AutoTokenizer


from opensora.models.causalvideovae import ae_stride_config, ae_channel_config, ae_norm, ae_denorm, CausalVAEModelWrapper
from opensora.models.diffusion.opensora.modeling_inpaint import OpenSoraInpaint
from opensora.models.diffusion.udit.modeling_udit import UDiTT2V

from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

from opensora.sample.pipeline_inpaint import OpenSoraInpaintPipeline

import imageio

# for validation
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda
from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop, LongSideResizeVideo, SpatialStrideCropVideo
import numpy as np
from einops import rearrange

def main(args):
    # torch.manual_seed(args.seed)
    # torch.backends.cuda.matmul.allow_tf32 = False
    weight_dtype = torch.bfloat16
    device = torch.device(args.device)

    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir)
    vae = CausalVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]

    norm_fun = Lambda(lambda x: 2. * x - 1.)

    resize = [CenterCropResizeVideo((args.height, args.width)), ]
    transform = transforms.Compose([
        ToTensorVideo(),
        *resize, 
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])

    def preprocess_images(images):
        print(images)
        if len(images) == 1:
            conditional_images_indices = [0]
        elif len(images) == 2:
            conditional_images_indices = [0, -1]
        else:
            print("Only support 1 or 2 condition images!")
            raise NotImplementedError
        
        try:
            conditional_images = [Image.open(image).convert("RGB") for image in images]
            conditional_images = [torch.from_numpy(np.copy(np.array(image))) for image in conditional_images]
            conditional_images = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in conditional_images]
            conditional_images = [transform(image).to(device=device, dtype=weight_dtype) for image in conditional_images]
        except Exception as e:
            print('Error when loading images')
            print(f'condition images are {images}')
            raise e
        return dict(conditional_images=conditional_images, conditional_images_indices=conditional_images_indices)


    transformer_model = OpenSoraInpaint.from_pretrained(args.model_path, cache_dir=args.cache_dir, 
                                                        low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)

    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler(clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler(clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        '''
        DPM++ 2M	        DPMSolverMultistepScheduler	
        DPM++ 2M Karras	    DPMSolverMultistepScheduler	init with use_karras_sigmas=True
        DPM++ 2M SDE	    DPMSolverMultistepScheduler	init with algorithm_type="sde-dpmsolver++"
        DPM++ 2M SDE Karras	DPMSolverMultistepScheduler	init with use_karras_sigmas=True and algorithm_type="sde-dpmsolver++"
        
        DPM++ SDE	        DPMSolverSinglestepScheduler	
        DPM++ SDE Karras	DPMSolverSinglestepScheduler	init with use_karras_sigmas=True
        DPM2	            KDPM2DiscreteScheduler	
        DPM2 Karras	        KDPM2DiscreteScheduler	init with use_karras_sigmas=True
        DPM2 a	            KDPM2AncestralDiscreteScheduler	
        DPM2 a Karras	    KDPM2AncestralDiscreteScheduler	init with use_karras_sigmas=True
        '''
        # scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
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
    pipeline = OpenSoraInpaintPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                transformer=transformer_model)
    pipeline.to(dtype=weight_dtype, device=device)
    if args.compile:
        # 5%  https://github.com/siliconflow/onediff/tree/main/src/onediff/infer_compiler/backends/nexfort
        options = '{"mode": "max-optimize:max-autotune:freezing:benchmark:low-precision",             \
                    "memory_format": "channels_last", "options": {"inductor.optimize_linear_epilogue": false, \
                    "triton.fuse_attention_allow_fp16_reduction": false}}'
        # options = '{"mode": "max-autotune", "memory_format": "channels_last",              \
        #             "options": {"inductor.optimize_linear_epilogue": false, "triton.fuse_attention_allow_fp16_reduction": false}}'
        from onediffx import compile_pipe
        pipeline = compile_pipe(
                pipeline, backend="nexfort", options=options, fuse_qkv_projections=True
            )

        # 4%
        # pipeline.transformer = torch.compile(pipeline.transformer)
    
    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)
        
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        text_prompt = [i.strip() for i in text_prompt]

    if not isinstance(args.conditional_images_path, list):
        args.conditional_images_path = [args.conditional_images_path]
    if len(args.conditional_images_path) == 1 and args.conditional_images_path[0].endswith('txt'):
        temp = open(args.conditional_images_path[0], 'r').readlines()
        conditional_images = [i.strip().split(',') for i in temp]
        
    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), 
    {}. 
    """
    
    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

    video_grids = []
    for idx, (prompt, images) in enumerate(zip(text_prompt, conditional_images)):

        pre_results = preprocess_images(images)
        cond_imgs = pre_results['conditional_images']
        cond_imgs_indices = pre_results['conditional_images_indices']

        videos = pipeline(
            conditional_images=cond_imgs,
            conditional_images_indices=cond_imgs_indices,
            prompt=positive_prompt.format(prompt),
            negative_prompt=negative_prompt, 
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            mask_feature=True,
            device=args.device, 
            max_sequence_length=args.max_sequence_length, 
        ).images
        try:
            if args.num_frames == 1:
                ext = 'jpg'
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_image(videos / 255.0, os.path.join(args.save_img_path, f'{idx}.{ext}'), nrow=1, normalize=True, value_range=(0, 1))  # t c h w
            else:
                ext = 'mp4'
                imageio.mimwrite(
                    os.path.join(args.save_img_path, f'{idx}.{ext}'), videos[0], fps=args.fps, quality=6)  # highest quality is 10, lowest is 0
            save_path = os.path.join(args.save_img_path, f'{idx}.{ext}')
            print(f'image or video is saved at {save_path}')
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    if args.num_frames == 1:
        save_image(video_grids / 255.0, os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                   nrow=math.ceil(math.sqrt(len(video_grids))), normalize=True, value_range=(0, 1))
    else:
        video_grids = save_video_grid(video_grids)
        imageio.mimwrite(os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'), video_grids, fps=args.fps, quality=6)

    print('save path {}'.format(args.save_img_path))


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
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--max_sequence_length", type=int, default=300)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.125)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true')

    parser.add_argument('--conditional_images_path', type=str, help='the path of txt file containing the paths of condition images')
    args = parser.parse_args()

    main(args)
