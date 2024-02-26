# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import math
import random

import numpy as np
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
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.max_image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.max_image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.max_image_size == 256
    else:
        learn_sigma = False

    # Load model:
    max_latent_size = args.max_image_size // 8
    model = SiT_models[args.model](
        input_size=max_latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.max_image_size}x{args.max_image_size}.pt"
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

    def pad_to_multiple(number, ds_stride):
        remainder = number % ds_stride
        if remainder == 0:
            return number
        else:
            padding = ds_stride - remainder
            return number + padding

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}", cache_dir='cache_dir').to(device)

    from torch.nn import functional as F
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)
    vae_stride = 8
    patch_size = 2
    temproal_size = 1
    num_frames = 16
    ds_stride = vae_stride * patch_size
    temproal_ds_stride = vae_stride * temproal_size

    # pad to max multiple of ds_stride
    batch_input_size = [[3, random.randint(1, num_frames), random.randint(1, args.max_image_size), random.randint(1, args.max_image_size)] for _ in range(n)]
    batch_images = [torch.rand(i) for i in batch_input_size]
    max_t, max_h, max_w = max([i[1] for i in batch_input_size]), max([i[2] for i in batch_input_size]), max([i[3] for i in batch_input_size])
    pad_max_t, pad_max_h, pad_max_w = pad_to_multiple(max_t, temproal_ds_stride), pad_to_multiple(max_h, ds_stride), pad_to_multiple(max_w, ds_stride)
    each_pad_t_h_w = [[pad_max_t - i.shape[1], pad_max_h - i.shape[2], pad_max_w - i.shape[3]] for i in batch_images]

    pad_batch_images = [F.pad(im, (0, pad_w, 0, pad_h, 0, pad_t), value=0) for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_images)]

    # make attention_mask
    args.max_image_size = [pad_max_t, pad_max_h, pad_max_w]
    max_latent_size = [args.max_image_size[0] // vae_stride, args.max_image_size[1] // vae_stride, args.max_image_size[2] // vae_stride]
    max_patchify_latent_size = [max_latent_size[0] // temproal_size, max_latent_size[1] // patch_size, max_latent_size[2] // patch_size]

    valid_patchify_latent_size = [[int(math.ceil(i[1]/temproal_ds_stride)),
                                   int(math.ceil(i[2]/ds_stride)),
                                   int(math.ceil(i[3]/ds_stride))] for i in batch_input_size]

    attention_mask = [F.pad(torch.ones(i), (0, max_patchify_latent_size[2]-i[2],
                                            0, max_patchify_latent_size[1]-i[1],
                                            0, max_patchify_latent_size[0]-i[0]), value=0) for i in valid_patchify_latent_size]
    attention_mask = torch.stack(attention_mask).to(device)
    # import ipdb
    # ipdb.set_trace()

    # return pad_batch_images, attention_mask

    assert max_latent_size[0] % temproal_size == 0 and max_latent_size[1] % patch_size == 0 and max_latent_size[2] % patch_size == 0
    print(valid_patchify_latent_size)
    print(max_latent_size)
    # Create sampling noise:
    z = torch.randn(n, 4, max_latent_size[0], max_latent_size[1], max_latent_size[2], device=device)  # latent_size=32
    # z = torch.randn(n, 4, 28, 30, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, attention_mask=attention_mask)

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    print(samples.shape)
    samples0 = vae.decode(samples[:, :, 0, :, :] / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")
    # Save and display images:
    save_image(samples0, f"sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    samples1 = vae.decode(samples[:, :, 1, :, :] / 0.18215).sample
    save_image(samples1, f"sample1.png", nrow=4, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    # parser.add_argument("--sampler-type", type=str, default="ODE", choices=["ODE", "SDE"])
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--max-image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--max-temproal-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")


    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
