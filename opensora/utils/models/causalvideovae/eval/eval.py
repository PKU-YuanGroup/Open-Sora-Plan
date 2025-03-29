import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
from glob import glob

sys.path.append(".")

from opensora.models.causalvideovae.eval.cal_lpips import calculate_lpips
from opensora.models.causalvideovae.eval.cal_fvd import calculate_fvd
from opensora.models.causalvideovae.eval.cal_psnr import calculate_psnr
from opensora.models.causalvideovae.eval.cal_ssim import calculate_ssim
from opensora.models.causalvideovae.dataset.video_dataset import (
    ValidVideoDataset,
    DecordInit,
    Compose,
    Lambda,
    resize,
    CenterCropVideo,
    ToTensorVideo
)

class EvalDataset(ValidVideoDataset):
    def __init__(
        self,
        real_video_dir,
        generated_video_dir,
        num_frames,
        sample_rate=1,
        crop_size=None,
        resolution=128,
    ) -> None:
        self.is_main_process = False
        self.v_decoder = DecordInit()
        self.real_video_files = []
        self.generated_video_files = self._make_dataset(generated_video_dir)
        for video_file in self.generated_video_files:
            filename = os.path.basename(video_file)
            if not os.path.exists(os.path.join(real_video_dir, filename)):
                raise Exception(os.path.join(real_video_dir, filename))
            self.real_video_files.append(os.path.join(real_video_dir, filename))
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.short_size = resolution
        self.transform = Compose(
            [
                ToTensorVideo(),
                Lambda(lambda x: resize(x, self.short_size)),
                (
                    CenterCropVideo(crop_size)
                    if crop_size is not None
                    else Lambda(lambda x: x)
                ),
            ]
        )

    def _make_dataset(self, real_video_dir):
        samples = []
        samples += sum(
            [
                glob(os.path.join(real_video_dir, f"*.{ext}"), recursive=True)
                for ext in self.video_exts
            ],
            [],
        )
        return samples
    
    def __len__(self):
        return len(self.real_video_files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_video_file = self.real_video_files[index]
        generated_video_file = self.generated_video_files[index]
        real_video_tensor = self._load_video(real_video_file, self.sample_rate)
        generated_video_tensor = self._load_video(generated_video_file, 1)
        return {"real": self.transform(real_video_tensor), "generated": self.transform(generated_video_tensor)}


def calculate_common_metric(args, dataloader, device):
    score_list = []
    for batch_data in tqdm(dataloader):
        real_videos = batch_data["real"].to(device)
        generated_videos = batch_data["generated"].to(device)

        assert real_videos.shape[2] == generated_videos.shape[2]
        if args.metric == "fvd":
            tmp_list = list(
                calculate_fvd(
                    real_videos, generated_videos, args.device, method=args.fvd_method
                )["value"].values()
            )
        elif args.metric == "ssim":
            tmp_list = list(
                calculate_ssim(real_videos, generated_videos)["value"].values()
            )
        elif args.metric == "psnr":
            tmp_list = [calculate_psnr(real_videos, generated_videos)]
        else:
            tmp_list = [calculate_lpips(real_videos, generated_videos, args.device)]
        score_list += tmp_list
    return np.mean(score_list)


def main():

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    dataset = EvalDataset(
        args.real_video_dir,
        args.generated_video_dir,
        num_frames=args.num_frames,
        sample_rate=args.sample_rate,
        crop_size=args.crop_size,
        resolution=args.resolution,
    )

    if args.subset_size:
        indices = range(args.subset_size)
        dataset = Subset(dataset, indices=indices)

    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=num_workers, pin_memory=True
    )

    metric_score = calculate_common_metric(args, dataloader, device)
    print(metric_score)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--real_video_dir", type=str, help=("the path of real videos`"))
    parser.add_argument(
        "--generated_video_dir", type=str, help=("the path of generated videos`")
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Like cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of processes to use for data loading. "
            "Defaults to `min(8, num_cpus)`"
        ),
    )
    parser.add_argument("--sample_fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=336)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default="fvd",
        choices=["fvd", "psnr", "ssim", "lpips", "flolpips"],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main()
