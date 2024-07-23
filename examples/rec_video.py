import math
import random
import argparse
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from decord import VideoReader, cpu
from torch.nn import functional as F
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Lambda, Compose
from opensora.models import CausalVAEModelWrapper
from opensora.dataset.transform import ToTensorVideo, CenterCropResizeVideo


def array_to_video(image_array: npt.NDArray, fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1).numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = random.randint(0, total_frames - sample_frames_len - 1)
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(f'sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}', video_path,
              total_frames)

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess(video_data: torch.Tensor, height: int = 128, width: int = 128) -> torch.Tensor:
    transform = Compose(
        [
            ToTensorVideo(),
            CenterCropResizeVideo((height, width)),
            Lambda(lambda x: 2. * x - 1.)
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


def main(args: argparse.Namespace):
    device = args.device
    kwarg = {}
    # vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir='cache_dir', **kwarg).to(device)
    vae = CausalVAEModelWrapper(args.ae_path, **kwarg).to(device)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
    vae.eval()
    vae = vae.to(device)
    vae = vae.half()

    with torch.no_grad():
        x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.height,
                           args.width)
        print(x_vae.shape)
        x_vae = x_vae.to(device, dtype=torch.float16)  # b c t h w
        latents = vae.encode(x_vae)
        print(latents.shape)
        latents = latents.to(torch.float16)
        video_recon = vae.decode(latents)  # b t c h w
        print(video_recon.shape)

    custom_to_video(video_recon[0], fps=args.fps, output_file=args.rec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--rec_path', type=str, default='')
    parser.add_argument('--ae', type=str, default='')
    parser.add_argument('--ae_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='results/pretrained')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--height', type=int, default=336)
    parser.add_argument('--width', type=int, default=336)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--sample_rate', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--tile_sample_min_size', type=int, default=256)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--save_memory', action='store_true')

    args = parser.parse_args()
    main(args)
