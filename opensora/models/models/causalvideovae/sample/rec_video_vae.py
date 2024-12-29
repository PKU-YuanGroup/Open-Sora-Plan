import argparse
from tqdm import tqdm
import torch
import sys
from torch.utils.data import DataLoader, Subset
import os
from accelerate import Accelerator

sys.path.append(".")
from opensora.models.causalvideovae.model import *
from opensora.models.causalvideovae.dataset.video_dataset import ValidVideoDataset
from opensora.models.causalvideovae.utils.video_utils import custom_to_video

@torch.no_grad()
def main(args: argparse.Namespace):
    accelerator = Accelerator()
    device = accelerator.device
    
    real_video_dir = args.real_video_dir
    generated_video_dir = args.generated_video_dir
    sample_rate = args.sample_rate
    resolution = args.resolution
    crop_size = args.crop_size
    num_frames = args.num_frames
    sample_rate = args.sample_rate
    device = args.device
    sample_fps = args.sample_fps
    batch_size = args.batch_size
    num_workers = args.num_workers
    subset_size = args.subset_size
    
    if not os.path.exists(args.generated_video_dir):
        os.makedirs(args.generated_video_dir, exist_ok=True)
    
    data_type = torch.bfloat16
    
    # ---- Load Model ----
    device = args.device
    model_cls = ModelRegistry.get_model(args.model_name)
    vae = model_cls.from_pretrained(args.from_pretrained)
    vae = vae.to(device).to(data_type)
    if args.enable_tiling:
        vae.enable_tiling()
        vae.tile_overlap_factor = args.tile_overlap_factor

    # ---- Prepare Dataset ----
    dataset = ValidVideoDataset(
        real_video_dir=real_video_dir,
        num_frames=num_frames,
        sample_rate=sample_rate,
        crop_size=crop_size,
        resolution=resolution,
    )
    if subset_size:
        indices = range(subset_size)
        dataset = Subset(dataset, indices=indices)
        
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers
    )
    dataloader = accelerator.prepare(dataloader)

    # ---- Inference ----
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        x, file_names = batch['video'], batch['file_name']
        x = x.to(device=device, dtype=data_type)  # b c t h w
        x = x * 2 - 1
        encode_result = vae.encode(x)
        if isinstance(encode_result, tuple):
            encode_result = encode_result[0]
        latents = encode_result.sample().to(data_type)
        video_recon = vae.decode(latents)
        if isinstance(video_recon, tuple):
            video_recon = video_recon[0]
        for idx, video in enumerate(video_recon):
            output_path = os.path.join(generated_video_dir, file_names[idx])
            if args.output_origin:
                os.makedirs(os.path.join(generated_video_dir, "origin/"), exist_ok=True)
                origin_output_path = os.path.join(generated_video_dir, "origin/", file_names[idx])
                custom_to_video(
                    x[idx], fps=sample_fps / sample_rate, output_file=origin_output_path
                )
            custom_to_video(
                video, fps=sample_fps / sample_rate, output_file=output_path
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_video_dir", type=str, default="")
    parser.add_argument("--generated_video_dir", type=str, default="")
    parser.add_argument("--from_pretrained", type=str, default="")
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--output_origin', action='store_true')
    parser.add_argument("--model_name", type=str, default=None, help="")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
    
