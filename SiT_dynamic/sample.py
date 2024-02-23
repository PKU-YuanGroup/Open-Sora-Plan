# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        # assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models["SiT-XL/2"](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
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
    
    if args.is_image:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        # Use Video VAE
        pass

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    
    # Create sampling noise:
    n = len(class_labels)
    if args.is_image:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
    else:
        z = torch.randn(n, 4, args.n_frame, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    if args.is_image:
        samples = vae.decode(samples / 0.18215).sample
        print(f"Sampling took {time() - start_time:.2f} seconds.")
        save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    else:
        # Use Video VAE
        pass 
        print(f"Sampling took {time() - start_time:.2f} seconds.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    # parser.add_argument("--sampler-type", type=str, default="ODE", choices=["ODE", "SDE"])
    
    # parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--is_image", type=bool, default=True)
    parser.add_argument("--n-frame", type=int, default=4)

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
