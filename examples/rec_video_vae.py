import random
import argparse
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from decord import VideoReader, cpu
from torch.nn import functional as F
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Lambda, Compose
from torchvision.transforms._transforms_video import RandomCropVideo
import sys
from tqdm import tqdm

sys.path.append(".")
from opensora.models.ae.videobase import CausalVAEModel
from opensora.models.ae.videobase.modules.conv import BlockwiseConv3d
import torch.nn as nn
import math

def array_to_video(
    image_array: npt.NDArray, fps: float = 30.0, output_file: str = "output_video.mp4"
) -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()


def custom_to_video(
    x: torch.Tensor, fps: float = 2.0, output_file: str = "output_video.mp4"
) -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    x = x.permute(1, 2, 3, 0).numpy()
    x = (255 * x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return


def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(
            f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
            video_path,
            total_frames,
        )

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess(
    video_data: torch.Tensor, short_size: int = 128, crop_size: Optional[int] = None
) -> torch.Tensor:

    transform = Compose(
        [
            Lambda(lambda x: ((x / 255.0) * 2 - 1)),
            ShortSideScale(size=short_size),
            (
                RandomCropVideo(size=crop_size)
                if crop_size is not None
                else Lambda(lambda x: x)
            ),
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


def replace_conv3d_with_blockwise(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv3d):
            new_module = BlockwiseConv3d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                (4, 64, 64),
            )
            new_module.set_weights(
                module.weight.data,
                module.bias.data if module.bias is not None else None,
            )
            setattr(model, name, new_module)
        else:
            replace_conv3d_with_blockwise(module)

def process_in_chunks(video_data: torch.Tensor, model: nn.Module, chunk_size: int, overlap: int, device: str):
    assert (chunk_size + overlap - 1) % 4 == 0
    num_frames = video_data.size(2)
    output_chunks = []
    
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        if start + chunk_size + overlap < num_frames:
            end += overlap
        chunk = video_data[:, :, start:end, :, :]
        
        with torch.no_grad():
            chunk = chunk.half().to(device)
            latents = model.encode(chunk)
            recon_chunk = model.decode(latents.sample().half()).cpu().float()
        
        if output_chunks:
            overlap_step = min(overlap, recon_chunk.shape[2])
            overlap_tensor = output_chunks[-1][:, :, -overlap_step:] * 1 / 4 + recon_chunk[:, :, :overlap_step] * 3 / 4
            output_chunks[-1] = torch.cat((output_chunks[-1][:, :, :-overlap], overlap_tensor), dim=2)
            if end < num_frames:
                output_chunks.append(recon_chunk[:, :, overlap:])
            else:
                output_chunks.append(recon_chunk[:, :, :, :, :])
        else:
            output_chunks.append(recon_chunk)
        start += chunk_size
    return torch.cat(output_chunks, dim=2)

def main(args: argparse.Namespace):
    video_path = args.video_path
    num_frames = args.num_frames
    resolution = args.resolution
    crop_size = args.crop_size
    sample_fps = args.sample_fps
    sample_rate = args.sample_rate

    device = args.device
    torch.cuda.reset_max_memory_allocated()
    start_memory = torch.cuda.memory_allocated() / 2**30
    vqvae = CausalVAEModel.load_from_checkpoint(args.ckpt)
    # replace_conv3d_with_blockwise(vqvae)
    vqvae = vqvae.half().to(device)
    with torch.no_grad():
        x_vae = preprocess(
            read_video(video_path, num_frames, sample_rate), resolution, crop_size
        )
        print(x_vae.shape)
        if x_vae.size(2) > 32:
            video_recon = process_in_chunks(x_vae, vqvae, chunk_size=4, overlap=1, device=device)
        else:
            x_vae = x_vae.half().to(device)
            latents = vqvae.encode(x_vae)
            video_recon = vqvae.decode(latents.sample().half())

    peak_memory_usage = torch.cuda.max_memory_allocated() / 2**30
    print(peak_memory_usage - start_memory)
    print(video_recon[0].shape)
    custom_to_video(
        video_recon[0], fps=sample_fps / sample_rate, output_file=args.rec_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default="")
    parser.add_argument("--rec-path", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--sample-fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
