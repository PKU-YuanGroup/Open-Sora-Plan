# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys

from accelerate import Accelerator
from tqdm import tqdm

from opensora.dataset import ae_denorm
from opensora.models.ae import ae_channel_config, getae, ae_stride_config
from opensora.models.ae.videobase import CausalVQVAEModelWrapper
from opensora.models.diffusion import Diffusion_models
from opensora.models.diffusion.diffusion import create_diffusion_T as create_diffusion
from opensora.models.diffusion.latte.modeling_latte import Latte
from opensora.utils.utils import find_model

import torch
import argparse

from einops import rearrange
import imageio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = (args.image_size // ae_stride_config[args.ae][1], args.image_size // ae_stride_config[args.ae][2])
    args.latent_size = latent_size
    model = Latte.from_pretrained(args.ckpt, subfolder="model").to(device)

    model.eval()  # important!

    model = accelerator.prepare(model)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    ae = getae(args).to(device)
    if isinstance(ae, CausalVQVAEModelWrapper):
        video_length = args.num_frames // ae_stride_config[args.ae][0] + 1
    else:
        video_length = args.num_frames // ae_stride_config[args.ae][0]
    bar = tqdm(range(args.num_sample))
    for i in bar:
        # Create sampling noise:
        z = torch.randn(1, model.module.in_channels, video_length, latent_size[0], latent_size[1], device=device)

        # Setup classifier-free guidance:
        if using_cfg and args.train_classcondition:
            z = torch.cat([z, z], 0)
            y = torch.randint(0, args.num_classes, (1,), device=device)
            cls_id = str(int(y.detach().cpu()))
            y_null = torch.tensor([args.num_classes] * 1, device=device)
            y = torch.cat([y, y_null], dim=0)
            model_kwargs = dict(class_labels=y, cfg_scale=args.cfg_scale)
            sample_fn = model.module.forward_with_cfg
        else:
            if args.train_classcondition:
                sample_fn = model.forward
                y = torch.randint(0, args.num_classes, (1,), device=device)
                cls_id = str(int(y.detach().cpu()))
                model_kwargs = dict(class_labels=y)
            else:
                sample_fn = model.forward
                model_kwargs = dict(class_labels=None)

        # Sample images:
        if args.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        elif args.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

        with torch.no_grad():
            samples = ae.decode(samples)
        # Save and display images:

        if not os.path.exists(args.save_video_path):
            os.makedirs(args.save_video_path)

        video_ = (ae_denorm[args.ae](samples[0]) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        if args.train_classcondition:
            video_save_path = os.path.join(args.save_video_path, f"sample_{i:03d}_cls" + str(cls_id) + '.mp4')
        else:
            video_save_path = os.path.join(args.save_video_path, f"sample_{i:03d}" + '.mp4')
        print(video_save_path)
        imageio.mimwrite(video_save_path, video_, fps=args.fps, quality=9)
        print('save path {}'.format(args.save_video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model", type=str, default='Latte-XL/122')
    parser.add_argument("--ae", type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=101)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--train_classcondition", action="store_true")
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sample_method", type=str, default='ddpm')
    parser.add_argument("--mixed_precision", type=str, default=None, choices=[None, "fp16", "bf16"])
    parser.add_argument("--attention_mode", type=str, choices=['xformers', 'math', 'flash'], default="math")
    args = parser.parse_args()
    main(args)
