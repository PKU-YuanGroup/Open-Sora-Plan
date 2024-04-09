import sys

sys.path.append(".")

from opensora.models.ae.videobase.dataset_videobase import VideoDataset
from opensora.models.ae.videobase import (
    VQVAEModel,
    VQVAEConfiguration,
    VQVAETrainer,
)
import argparse
from typing import Optional
from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field, asdict


@dataclass
class VQVAEArgument:
    embedding_dim: int = (field(default=256),)
    n_codes: int = (field(default=2048),)
    n_hiddens: int = (field(default=240),)
    n_res_layers: int = (field(default=4),)
    resolution: int = (field(default=128),)
    sequence_length: int = (field(default=16),)
    downsample: str = (field(default="4,4,4"),)
    no_pos_embd: bool = (True,)
    data_path: str = field(default=None, metadata={"help": "data path"})


@dataclass
class VQVAETrainingArgument(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
    )


def train(args, vqvae_args: VQVAEArgument, training_args: VQVAETrainingArgument):
    # Load Config
    config = VQVAEConfiguration(
        embedding_dim=vqvae_args.embedding_dim,
        n_codes=vqvae_args.n_codes,
        n_hiddens=vqvae_args.n_hiddens,
        n_res_layers=vqvae_args.n_res_layers,
        resolution=vqvae_args.resolution,
        sequence_length=vqvae_args.sequence_length,
        downsample=vqvae_args.downsample,
        no_pos_embd=vqvae_args.no_pos_embd,
    )
    # Load Model
    model = VQVAEModel(config)
    # Load Dataset
    dataset = VideoDataset(
        args.data_path,
        sequence_length=args.sequence_length,
        resolution=config.resolution,
    )
    # Load Trainer
    trainer = VQVAETrainer(model, training_args, train_dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((VQVAEArgument, VQVAETrainingArgument))
    vqvae_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(vqvae_args), **vars(training_args))
    set_seed(args.seed)

    train(args, vqvae_args, training_args)
