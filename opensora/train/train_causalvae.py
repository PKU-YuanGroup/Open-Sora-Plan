import sys
sys.path.append(".")

from opensora.models.ae.videobase import (
    CausalVAEConfiguration,
    CausalVAEDataset,
    CausalVAEModel,
    CausalVAETrainer,
)
import argparse
from typing import Optional
from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field, asdict
from typing import Tuple, List
import torch.distributed as dist
import os
import cv2
import torch
import numpy as np

def array_to_video(image_array, fps: float = 30.0, output_file: str = 'output_video.mp4') -> None:
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
    array_to_video(x, fps=fps, output_file=output_file)
    # imageio.mimwrite(output_file, x, fps=fps, quality=9)
    return


@dataclass
class DataArguments:
    data_path: str = field(default="./UCF101")
    video_num_frames: int = field(default=17)
    sample_rate: int = field(default=1)

@dataclass
class CausalVAEArguments:
    in_channels: int = field(default=3)
    out_channels: int = field(default=3)
    hidden_size: int = field(default=128)
    z_channels: int = field(default=16)
    ch_mult: Tuple[int] = field(default=(1, 1, 2, 2, 4))
    num_res_block: int = field(default=2)
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    dropout: float = field(default=0.0)
    resolution: int = field(default=256)
    attn_type: str = field(default="vanilla3D")
    use_linear_attn: bool = field(default=False)
    embed_dim: int = field(default=16)
    time_compress: int = field(default=2)
    logvar_init: float = field(default=0.0)
    kl_weight: float = field(default=1e-6)
    pixelloss_weight: int = field(default=1)
    perceptual_weight: int = field(default=1)
    disc_loss: str = field(default="hinge")

@dataclass
class CausalVAETrainingArguments(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

def train(args, vqvae_args, training_args):
    # Load Config
    config = CausalVAEConfiguration(**asdict(vqvae_args))
    # Load Model
    if training_args.resume_from_checkpoint is not None:
        model = CausalVAEModel.load_from_checkpoint(training_args.resume_from_checkpoint)
    else:
        model = CausalVAEModel(config)
    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        print(model)
    # Load Dataset
    dataset = CausalVAEDataset(args.data_path, sequence_length=args.video_num_frames, resolution=config.resolution, sample_rate=args.sample_rate)
    custom_to_video(dataset[0]['video'], fps=10, output_file="dataset.mp4")
    # Load Trainer
    trainer = CausalVAETrainer(model, training_args, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    parser = HfArgumentParser((CausalVAEArguments, CausalVAETrainingArguments, DataArguments))
    
    if not sys.argv[-1].endswith(".yaml"):
        raise Exception("Please use yaml config.")
    
    vqvae_args, training_args, data_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    args = argparse.Namespace(**vars(vqvae_args), **vars(training_args), **vars(data_args))
    set_seed(args.seed)
    train(args, vqvae_args, training_args)
