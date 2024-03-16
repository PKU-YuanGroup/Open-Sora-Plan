import sys
sys.path.append(".")

from opensora.models.ae.videobase import (
    CausalVQVAEModel,
    CausalVQVAEConfiguration,
    CausalVQVAEDataset,
    CausalVQVAETrainer,
)

import argparse
from typing import Optional
from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field, asdict


@dataclass
class CausalVQVAEArgument:
    embedding_dim: int = field(default=256),
    n_codes: int = field(default=2048),
    n_hiddens: int = field(default=240),
    n_res_layers: int = field(default=4),
    resolution: int = field(default=128),
    sequence_length: int = field(default=16),
    time_downsample: int = field(default=4),
    spatial_downsample: int = field(default=8),
    no_pos_embd: bool = True,
    video_data_path: str = field(default=None, metadata={"help": "data path"})
    image_data_path: str = field(default=None, metadata={"help": "not implemented"})

@dataclass
class CausalVQVAETrainingArgument(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

def train(args, vqvae_args: CausalVQVAEArgument, training_args: CausalVQVAETrainingArgument):
    # Load Config
    config = CausalVQVAEConfiguration(
        embedding_dim=vqvae_args.embedding_dim,
        n_codes=vqvae_args.n_codes,
        n_hiddens=vqvae_args.n_hiddens,
        n_res_layers=vqvae_args.n_res_layers,
        resolution=vqvae_args.resolution,
        sequence_length=vqvae_args.sequence_length,
        time_downsample=vqvae_args.time_downsample,
        spatial_downsample=vqvae_args.spatial_downsample,
        no_pos_embd=vqvae_args.no_pos_embd
    )
    # Load Model
    if args.resume_from_checkpoint:
        model = CausalVQVAEModel.load_from_checkpoint(args.resume_from_checkpoint)
    else:
        model = CausalVQVAEModel(config)
    # Load Dataset
    dataset = CausalVQVAEDataset(args.video_data_path, image_folder=None, sequence_length=args.sequence_length, resolution=config.resolution)
    # Load Trainer
    trainer = CausalVQVAETrainer(model, training_args, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()

if __name__ == "__main__":
    parser = HfArgumentParser((CausalVQVAEArgument, CausalVQVAETrainingArgument))
    vqvae_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(vqvae_args), **vars(training_args))
    set_seed(args.seed)

    train(args, vqvae_args, training_args)
