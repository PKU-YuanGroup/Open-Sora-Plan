# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import os
import sys

from opensora.dataset import ae_denorm
from opensora.models.ae import ae_channel_config, getae, ae_stride_config
from opensora.models.diffusion import Diffusion_models
from opensora.models.diffusion.transport import create_transport, Sampler
from opensora.utils.utils import find_model

import torch
import argparse

from einops import rearrange
import imageio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def main(mode, args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    using_cfg = args.cfg_scale > 1.0
    
    # Load model:
    latent_size = args.image_size // ae_stride_config[args.ae][1]
    args.latent_size = latent_size
    model = Diffusion_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=ae_channel_config[args.ae],
        extras=args.extras
    ).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()  # important!
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )      
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    ae = getae(args).to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        ae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)

    # Labels to condition the model with (feel free to change):
    
    # Create sampling noise:
    if args.use_fp16:
        z = torch.randn(1, args.num_frames // ae_stride_config[args.ae][0], model.in_channels, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
    else:
        z = torch.randn(1, args.num_frames // ae_stride_config[args.ae][0], model.in_channels, latent_size, latent_size, device=device)

    # Setup classifier-free guidance:
    if using_cfg:
        z = torch.cat([z, z], 0)
        y = torch.randint(0, args.num_classes, (1,), device=device)
        y_null = torch.tensor([args.num_classes] * 1, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
        forward_fn = model.forward_with_cfg
    else:
        forward_fn = model.forward
        model_kwargs = dict(y=None, use_fp16=args.use_fp16) 
    
    # Sample images:
    samples = sample_fn(z, forward_fn, **model_kwargs)[-1]
    
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)
    samples = ae.decode(samples)

    # Save and display images:
    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)


    video_ = (ae_denorm[args.ae](samples[0]) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    video_save_path = os.path.join(args.save_video_path, 'sample' + '.mp4')
    print(video_save_path)
    imageio.mimwrite(video_save_path, video_, fps=args.fps, quality=9)
    print('save path {}'.format(args.save_video_path))


def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model", type=str, default='Latte-XL/122')
    parser.add_argument("--ae", type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument("--save-video-path", type=str, default="./sample_videos/")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--extras", type=int, default=1)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--sample-method", type=str, default='ddpm')

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
