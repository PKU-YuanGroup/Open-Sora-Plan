import sys
sys.path.append(".")
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, Lambda
from torch.nn import functional as F
import argparse
import numpy as np
from opensora.models.causalvideovae import ae_wrapper

def preprocess(video_data: torch.Tensor, short_size: int = 128) -> torch.Tensor:
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: 2. * x - 1.), 
            Resize(size=short_size),
        ]
    )
    outputs = transform(video_data)
    outputs = outputs.unsqueeze(0).unsqueeze(2)
    return outputs

def main(args: argparse.Namespace):
    image_path = args.image_path
    short_size = args.short_size
    device = args.device
    kwarg = {}
    
    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir', **kwarg).to(device)
    vae = ae_wrapper[args.ae](args.ae_path, **kwarg).eval().to(device)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.eval()
    vae = vae.to(device)
    vae = vae.half()
    
    with torch.no_grad():
        x_vae = preprocess(Image.open(image_path), short_size)
        x_vae = x_vae.to(device, dtype=torch.float16)  # b c t h w
        latents = vae.encode(x_vae)
        latents = latents.to(torch.float16)
        image_recon = vae.decode(latents)  # b t c h w
    x = image_recon[0, 0, :, :, :]
    x = x.squeeze()
    x = x.detach().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    x = (255*x).astype(np.uint8)
    x = x.transpose(1,2,0)
    image = Image.fromarray(x)
    image.save(args.rec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--rec_path', type=str, default='')
    parser.add_argument('--ae', type=str, default='')
    parser.add_argument('--ae_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='results/pretrained')
    parser.add_argument('--short_size', type=int, default=336)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    
    args = parser.parse_args()
    main(args)
