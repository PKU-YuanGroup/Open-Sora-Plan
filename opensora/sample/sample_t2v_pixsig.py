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
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

import os, sys

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V_S_122
from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_opensora import OpenSoraPipeline

import imageio


def main(args):
    # torch.manual_seed(args.seed)
    weight_dtype = torch.float16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # transformer_model = Transformer2DModel.from_pretrained(
    #     "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    #     subfolder='transformer', cache_dir=args.cache_dir,
    #     torch_dtype=weight_dtype,
    #     use_safetensors=True, low_cpu_mem_usage=True
    # )
    latent_size = (64, 64)
    latent_size_t = 1
    transformer_model = OpenSoraT2V_S_122(in_channels=4, 
                              out_channels=8, 
                              sample_size=latent_size, 
                              sample_size_t=latent_size_t, 
                              activation_fn="gelu-approximate",
                              attention_bias=True,
                              attention_type="default",
                              double_self_attention=False,
                              norm_elementwise_affine=False,
                              norm_eps=1e-06,
                              norm_num_groups=32,
                              num_vector_embeds=None,
                              only_cross_attention=False,
                              upcast_attention=False,
                              use_linear_projection=False,
                              use_additional_conditions=False).to(dtype=weight_dtype)
    print(2)
    path = "/remote-home1/yeyang/dev3d/Open-Sora-Plan/cache_dir/models--PixArt-alpha--PixArt-Sigma-XL-2-512-MS/snapshots/786c445c97ddcc0eb2faa157b131ac71ee1935a2/transformer/diffusion_pytorch_model.safetensors"
    from safetensors.torch import load_file as safe_load
    ckpt = safe_load(path, device="cpu")
    transformer_model.load_state_dict(ckpt)
    print(args.text_encoder_name)
    text_encoder = T5EncoderModel.from_pretrained('/remote-home1/yeyang/dev3d/Open-Sora-Plan/cache_dir/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37', low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained('/remote-home1/yeyang/dev3d/Open-Sora-Plan/cache_dir/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37', cache_dir=args.cache_dir)
    print(3)
    vae = AutoencoderKL.from_pretrained("/remote-home1/yeyang/dev3d/Open-Sora-Plan/vae", torch_dtype=torch.float16)
    vae.vae_scale_factor = (4, 8, 8)
    print(1)
    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler()
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
        
    pipeline = OpenSoraPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                transformer=transformer_model)
    pipeline.to(device)
    prompt = "A small cactus with a happy face in the Sahara desert."
    image = pipeline(prompt,
                    num_frames=1,
                    height=512,
                    width=512,
                    num_inference_steps=20,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    mask_feature=True,
                    ).images[0]
    image.save("./catcus.png")
    image = pipeline(prompt,
                    num_frames=1,
                    height=512,
                    width=512,
                    num_inference_steps=50,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                    mask_feature=True,
                    ).images[0]
    image.save("./catcus50.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default=None, choices=[None, '65x512x512', '65x256x256', '17x256x256'])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="DPMSolverMultistep")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    args = parser.parse_args()

    main(args)