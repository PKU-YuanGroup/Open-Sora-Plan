import random

import cv2
import numpy as np
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
import yaml
import torch
from omegaconf import OmegaConf
from ldm.models.autoencoder import AutoencoderKL

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vae(config, ckpt_path=None):
    model = AutoencoderKL(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(missing, unexpected)
    return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def array_to_video(image_array, fps=30, output_file='output_video.mp4'):
    height, width, channels = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, float(fps), (width, height))

    for image in image_array:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_rgb)

    video_writer.release()

def custom_to_video(x, fps=2, output_file='output_video.mp4'):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1, 2, 3, 0).numpy()  # (C, T, H, W) -> (T, H, W, C)
    x = (255*x).astype(np.uint8)
    array_to_video(x, fps=fps, output_file=output_file)
    return

def read_video(video_path, num_frames, sampling_fps, resampling_fps=30):
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)

    sampling_duration = num_frames / sampling_fps
    sampling_total_frames = int(sampling_duration * resampling_fps)
    # s = random.randint(0, total_frames - sampling_total_frames - 1)
    s = 0
    e = s + sampling_total_frames
    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data

def preprocess(video_data, short_size=128, crop_size=128):

    transform = Compose(
        [
            # UniformTemporalSubsample(num_frames),
            Lambda(lambda x: (2 * (x / 255.0) - 1)),
            # NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ShortSideScale(size=short_size),
            RandomCropVideo(size=crop_size),
            # RandomHorizontalFlipVideo(p=0.5),
        ]
    )

    video_outputs = transform(video_data)
    video_outputs = torch.unsqueeze(video_outputs, 0)

    return video_outputs


def reconstruct_with_vae(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z = model.encode(x)
    z_sample = z.sample()
    print(f"VAE --- {model.__class__.__name__}: latent shape: {z_sample.shape[1:]}")
    xrec = model.decode(z_sample)
    return xrec

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

config32x32 = load_config("latent-diffusion/configs/autoencoder/videoautoencoder_kl_32x32x4.yaml", display=False)
model32x32 = load_vae(config32x32, ckpt_path="last.ckpt").to(DEVICE)


num_frames, sampling_fps = 16, 6
short_size, crop_size = 128, 128
video_path = r"G:\instruction datasets\train_image_video\valley\000051_000100\1066650544.mp4"
x_vae = preprocess(read_video(video_path, num_frames, sampling_fps), short_size, crop_size)
x_vae = x_vae.to(DEVICE)

print(f"input is of size: {x_vae.shape}")

custom_to_video(x_vae[0], fps=sampling_fps, output_file='origin.mp4')

x0 = reconstruct_with_vae(x_vae, model32x32)
custom_to_video(x0[0], fps=sampling_fps, output_file='decode.mp4')

print(model32x32)

print(sum(p.numel() for p in model32x32.encoder.parameters()))
print(sum(p.numel() for p in model32x32.decoder.parameters()))

