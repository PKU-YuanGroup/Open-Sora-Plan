import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math

from accelerate import Accelerator
from accelerate.utils import set_seed
from opensora.eval.inversion.inversion_dataset import InversionEvalImageDataset
from opensora.models.causalvideovae.eval.cal_psnr import calculate_psnr

def calculate_mse_(videos1, videos2):
    assert videos1.shape == videos2.shape
    mse = (videos1 - videos2) ** 2
    mse = mse.mean(dim=list(range(1, mse.ndim)))
    return mse.cpu().numpy()

def calculate_psnr_(videos1, videos2):
    
    def img_psnr(img1, img2):
        # [0,1]
        # compute mse
        # mse = np.mean((img1-img2)**2)
        mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
        # compute psnr
        if mse < 1e-10:
            return 100
        psnr = 20 * math.log10(1 / math.sqrt(mse))
        return psnr

    psnr_results = []
    
    for video_num in range(videos1.shape[0]):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].cpu().numpy()
            img2 = video2[clip_timestamp].cpu().numpy()
            
            # calculate psnr of a video
            psnr_results_of_a_video.append(img_psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)
    
    psnr_results = np.array(psnr_results) # [batch_size, num_frames]
    return psnr_results

eval_fun = {
    'mse': calculate_mse_, 
    'psnr': calculate_psnr_, 
}
def main(args):
    # Prepare environment
    accelerator = Accelerator()
    set_seed(args.seed, device_specific=False)  # every process has the same seed
    device = accelerator.device
    # Prepare dataset
    dataset = InversionEvalImageDataset(args)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False
    )

    # Prepare accelerator
    dataloader = accelerator.prepare(dataloader)

    # Run pipeline
    final_inversion_loss = {
        m: {num_inverse_steps: [] for num_inverse_steps in args.num_inverse_steps} for m in args.metrics
            }

    for batch in tqdm(dataloader):
        # batch # [b, 1+num_inverse_steps, 1 c h w], [0, 1]
        batch = batch.float()
        bs = batch.shape[0]
        gt_batch = batch[:, :1].repeat(1, len(args.num_inverse_steps), 1, 1, 1, 1)  # [b, 1, 1 c h w] -> [b, num_inverse_steps, 1 c h w]
        inverse_batch = batch[:, 1:]  # [b, num_inverse_steps, 1 c h w]
        gt_batch = rearrange(gt_batch, "b n t c h w -> (b n) t c h w")
        inverse_batch = rearrange(inverse_batch, "b n t c h w -> (b n) t c h w")

        for m in args.metrics:
            score = eval_fun[m](gt_batch, inverse_batch)
            score = score.reshape(bs, -1)  # b n
            for i, num_inverse_steps in enumerate(args.num_inverse_steps):
                final_inversion_loss[m][num_inverse_steps].append(
                    score[:, i]
                )

    accelerator.wait_for_everyone()
    final_inversion_loss = (
        accelerator.gather_for_metrics(final_inversion_loss)
    )
    if accelerator.is_main_process:
        print("=" * 50)
        print("Inversion Loss Results")
        print("=" * 50)
        for m in args.metrics:
            print(f"Steps | {m.upper()}")
            print("-" * 50)
            for num_inverse_steps in args.num_inverse_steps:
                score = np.mean(final_inversion_loss[m][num_inverse_steps])
                print(f"{num_inverse_steps:5d} | {score:.6f}")
            print("=" * 50)
    return final_inversion_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="", help="data txt path")
    parser.add_argument("--data_root", type=str, default="", help="data txt path")
    parser.add_argument(
        "--num_inverse_steps",
        type=int,
        nargs="+",
        default=[10, 30, 50, 70],
        help="num inverse steps",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="num workers")
    parser.add_argument("--seed", type=int, default=1234, help="num workers")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[
            'mse', 
            'psnr', 
            'lpips', 
            'ssim'
            ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    import json
    args = parse_args()
    args.data_file = "/storage/dataset/inversion_data/in-domain/data/flower102/test.json"
    args.data_root = "/storage/ongoing/12.29/eval/t2i_ablation_arch_gen_inversion_nocondi/mixnorm/flowers102/test"
    args.num_workers = 8 
    args.num_inverse_steps = [40, 80, 90, 100]
    metrics = [
            'psnr', 
            'mse', 
        ]
    num_samples = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    final_inversion = {i: [] for i in num_samples}
    for m in metrics:
        args.metrics = [m]
        for i in num_samples:
            args.num_samples = i
            for k in range(10):
                args.seed = i*10 + k
                final_inversion_loss = main(args)
                final_inversion[i].append([float(np.array(final_inversion_loss[m][j]).mean()) for j in args.num_inverse_steps])
        with open(f'flower_inv_{m}.json', 'w') as f:
            json.dump(final_inversion, f, indent=2)