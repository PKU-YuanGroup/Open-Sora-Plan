import random
import argparse
from typing import Optional

import cv2
import imageio
import numpy as np
import numpy.typing as npt
import torch
from decord import VideoReader, cpu
from torch.nn import functional as F
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Lambda, Compose
from torchvision.transforms._transforms_video import RandomCropVideo

import sys
sys.path.append(".")
from opensora.models.ae import VQVAEModel


def array_to_video(image_array: npt.NDArray, fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # type: ignore
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()

def custom_to_video(x: torch.Tensor, fps: float = 2.0, output_file: str = 'output_video.mp4') -> None:
    x = x.detach().cpu()
    x = torch.clamp(x, -0.5, 0.5)
    x = (x + 0.5)
    x = x.permute(1, 2, 3, 0).numpy()  # (C, T, H, W) -> (T, H, W, C)
    x = (255*x).astype(np.uint8)
    # array_to_video(x, fps=fps, output_file=output_file)
    imageio.mimwrite(output_file, x, fps=fps, quality=9)
    return

def read_video(video_path: str, num_frames: int, sample_rate: int) -> torch.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = random.randint(0, total_frames - sample_frames_len - 1)
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

def preprocess(video_data: torch.Tensor, short_size: int = 128, crop_size: Optional[int] = None) -> torch.Tensor:

    transform = Compose(
        [
            # UniformTemporalSubsample(num_frames),
            Lambda(lambda x: ((x / 255.0) - 0.5)),
            # NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ShortSideScale(size=short_size),
            RandomCropVideo(size=crop_size) if crop_size is not None else Lambda(lambda x: x),
            # RandomHorizontalFlipVideo(p=0.5),
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


def main(args: argparse.Namespace):
    video_path = args.video_path
    num_frames = args.num_frames
    resolution = args.resolution
    crop_size = args.crop_size
    sample_fps = args.sample_fps
    sample_rate = args.sample_rate
    device = torch.device('cuda')
    if args.ckpt in ['bair_stride4x2x2', 'ucf101_stride4x4x4', 'kinetics_stride4x4x4', 'kinetics_stride2x4x4']:
        vqvae = VQVAEModel.download_and_load_model(args.ckpt)
    else:
        vqvae = VQVAEModel.load_from_checkpoint(args.ckpt)
    vqvae.eval()
    vqvae = vqvae.to(device)

    with torch.no_grad():
        x_vae = preprocess(read_video(video_path, num_frames, sample_rate), resolution, crop_size)
        x_vae = x_vae.to(device)
        encodings, embeddings = vqvae.encode(x_vae, include_embeddings=True)
        video_recon = vqvae.decode(encodings)

    # custom_to_video(x_vae[0], fps=sample_fps/sample_rate, output_file='origin_input.mp4')
    custom_to_video(video_recon[0], fps=sample_fps/sample_rate, output_file=args.rec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='')
    parser.add_argument('--rec-path', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='ucf101_stride4x4x4')
    parser.add_argument('--sample-fps', type=int, default=30)
    parser.add_argument('--resolution', type=int, default=336)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--sample-rate', type=int, default=1)
    args = parser.parse_args()
    main(args)
