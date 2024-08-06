import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import math
import os
from accelerate.utils import set_seed
import pip
import torch
import argparse
import torchvision

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, MT5EncoderModel, UMT5EncoderModel, AutoTokenizer


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.opensora.modeling_inpaint import OpenSoraInpaint, ModelType

from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.sample.pipeline_inpaint import hacked_pipeline_call, decode_latents
# for validation
import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda
from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo, TemporalRandomCrop, LongSideResizeVideo, SpatialStrideCropVideo, ToTensorAfterResize
import numpy as np
from einops import rearrange

import imageio
import glob
import gc
import time

@torch.inference_mode()
def validation(args):

    # torch.manual_seed(args.seed)
    weight_dtype = torch.bfloat16
    device = torch.device(f'cuda:{args.rank}')

    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir)
    vae = getae_wrapper(args.ae)(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]


    
    transformer_model = OpenSoraInpaint.from_pretrained(args.model_path, cache_dir=args.cache_dir, 
                                                        low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
    
    text_encoder = MT5EncoderModel.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir)
    

    transformer_model = transformer_model.to(device)
    vae.vae = vae.vae.to(device)
    text_encoder = text_encoder.to(device)

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
    # Save the generated videos
    save_dir = args.save_img_path
    os.makedirs(save_dir, exist_ok=True)

    pipeline = OpenSoraPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                transformer=transformer_model)

    pipeline.__call__ = hacked_pipeline_call.__get__(pipeline, OpenSoraPipeline)
    pipeline.decode_latents = decode_latents.__get__(pipeline, OpenSoraPipeline)

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

    positive_prompt = "(masterpiece), (best quality), (ultra-detailed), {}. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
    negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, 
                        """
    
    norm_fun = Lambda(lambda x: 2. * x - 1.)

    resize_transform = CenterCropResizeVideo((args.height, args.width))
    transform = transforms.Compose([
        ToTensorAfterResize(),
        # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
        norm_fun
    ])


    def preprocess_images(images):
        if len(images) == 1:
            condition_images_indices = [0]
        elif len(images) == 2:
            condition_images_indices = [0, -1]
        condition_images = [Image.open(image).convert("RGB") for image in images]
        condition_images = [torch.from_numpy(np.copy(np.array(image))) for image in condition_images]
        condition_images = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in condition_images]
        condition_images = [resize_transform(image) for image in condition_images]
        condition_images = [transform(image).to(device=device, dtype=weight_dtype) for image in condition_images]
        return dict(condition_images=condition_images, condition_images_indices=condition_images_indices)
    

    videos = []
    max_sample_num = args.max_sample_num // args.world_size
    current_sample_num = 0
    for idx, (prompt, images) in enumerate(zip(validation_prompt, validation_images_list)):

        if (current_sample_num + 1) > max_sample_num:
            break


        if not isinstance(images, list):
            images = [images]
        if 'img' in images[0]:
            continue

        if idx % args.world_size != args.rank:
            continue
        
        pre_results = preprocess_images(images)
        condition_images = pre_results['condition_images']
        condition_images_indices = pre_results['condition_images_indices']

        video = pipeline.__call__(
            prompt=prompt,
            condition_images=condition_images,
            condition_images_indices=condition_images_indices,
            negative_prompt=negative_prompt,
            vip_tokens=None,
            vip_attention_mask=None,
            negative_vip_tokens=None,
            negative_vip_attention_mask=None,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            mask_feature=True,
            device=device,
            max_sequence_length=args.max_sequence_length,
            model_type=ModelType.INPAINT_ONLY,
        ).images
        videos.append(video[0])

        ext = 'mp4'
        imageio.mimwrite(
            os.path.join(save_dir, f'{idx}.{ext}'), video[0], fps=24, quality=6)  # highest quality is 10, lowest is 0
        current_sample_num += 1


    dist.barrier()       
    video_grids = torch.stack(videos, dim=0).to(device=device)
    shape = list(video_grids.shape)
    shape[0] *= world_size
    gathered_tensor = torch.zeros(shape, dtype=video_grids.dtype, device=device)
    dist.all_gather_into_tensor(gathered_tensor, video_grids.contiguous())
    video_grids = gathered_tensor.cpu()

    if args.rank == 0:
    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
        if args.num_frames == 1:
            save_image(video_grids / 255.0, os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                    nrow=math.ceil(math.sqrt(len(video_grids))), normalize=True, value_range=(0, 1))
        else:
            video_grids = save_video_grid(video_grids)
            imageio.mimwrite(os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'), video_grids, fps=args.fps, quality=8)

        print('save path {}'.format(args.save_img_path))

    del pipeline
    del text_encoder
    del vae
    del transformer_model
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    lask_ckpt = None
    root_model_path = args.model_path
    root_save_path = args.save_img_path
 
    while True:
        # Get the most recent checkpoint
        dirs = os.listdir(root_model_path)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        dist.barrier()
        if path != lask_ckpt:
            print("====================================================")
            print(f"sample {path}...")
            args.model_path = os.path.join(root_model_path, path, "model")
            args.save_img_path = os.path.join(root_save_path, f"{path}_normal")
            print(f"model path: {args.model_path}, save img path: {args.save_img_path}")
            while True:
                if os.path.exists(args.model_path):
                    validation(args)
                    break
                else:
                    time.sleep(5)
                    continue
            print("====================================================")
            print(f"sample ema {path}...")
            args.model_path = os.path.join(root_model_path, path, "model_ema")
            args.save_img_path = os.path.join(root_save_path, f"{path}_ema")
            print(f"model path: {args.model_path}, save img path: {args.save_img_path}")
            while True:
                if os.path.exists(args.model_path):
                    validation(args)
                    break
                else:
                    time.sleep(5)
                    continue
                
            lask_ckpt = path
            
        else:
            print("no new ckpt, sleeping...")
        time.sleep(60)

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
    parser.add_argument('--model_3d', action='store_true')
    parser.add_argument('--enable_stable_fp32', action='store_true')


    parser.add_argument("--max_sample_num", type=int, default=8)
    parser.add_argument("--validation_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    args.world_size = world_size = torch.cuda.device_count()
    args.rank = int(os.environ["LOCAL_RANK"])

    # Initialize the process group
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)

    main(args)