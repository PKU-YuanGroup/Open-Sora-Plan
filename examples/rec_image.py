import sys
sys.path.append(".")
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torch.nn import functional as F
from opensora.models.ae.videobase import CausalVAEModel
import argparse
import numpy as np

def preprocess(video_data: torch.Tensor, short_size: int = 128) -> torch.Tensor:
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.5), (0.5)),
            Resize(size=short_size),
        ]
    )
    outputs = transform(video_data)
    outputs = outputs.unsqueeze(0).unsqueeze(2)
    return outputs

def main(args: argparse.Namespace):
    image_path = args.image_path
    resolution = args.resolution
    device = args.device
    
    vqvae = CausalVAEModel.load_from_checkpoint(args.ckpt)
    vqvae.eval()
    vqvae = vqvae.to(device)
    
    with torch.no_grad():
        x_vae = preprocess(Image.open(image_path), resolution)
        x_vae = x_vae.to(device)
        latents = vqvae.encode(x_vae)
        recon = vqvae.decode(latents.sample())
    x = recon[0, :, 0, :, :]
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
    parser.add_argument('--image-path', type=str, default='')
    parser.add_argument('--rec-path', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--resolution', type=int, default=336)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)
