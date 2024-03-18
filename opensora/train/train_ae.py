import argparse
from typing import Optional
from dataclasses import dataclass, field, asdict

from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments

import sys
sys.path.append(".")

from opensora.models.ae.videobase import (
    CausalVQVAEModel, CausalVQVAEConfiguration,
    VQVAEModel, VQVAEConfiguration,
    build_videoae_dataset, VideoAETrainer
)


@dataclass
class AEArgument:
    embedding_dim: int = field(default=256)
    n_codes: int = field(default=2048)
    n_hiddens: int = field(default=240)
    n_res_layers: int = field(default=4)
    resolution: int = field(default=128)
    sequence_length: int = field(default=16)
    # argument for vqvae
    downsample: str = field(default="4,4,4")
    # argument for causal vqvae
    time_downsample: int = field(default=4)
    spatial_downsample: int = field(default=8)
    no_pos_embd: bool = True,
    video_data_path: str = field(default=None, metadata={"help": "data path"})
    image_data_path: str = field(default=None, metadata={"help": "not implemented"})


@dataclass
class AETrainingArgument(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    model_name: Optional[str] = field(
        default="vqvae", metadata={"help": "train model"}
    )
    data_type: Optional[str] = field(
        default="ae", metadata={"help": "data type"}
    )


def train(args, ae_args, training_args):
    # Load Config
    if training_args.model_name == "vqvae":
        model_class = VQVAEModel
        config = VQVAEConfiguration(**asdict(ae_args))
    elif training_args.model_name == "causalvqvae":
        model_class = CausalVQVAEModel
        config = CausalVQVAEConfiguration(**asdict(ae_args))
    else:
        raise ValueError(f"Invalid model class: {training_args.model_name}")

    # Load Model
    if args.resume_from_checkpoint:
        model = model_class.load_from_checkpoint(args.resume_from_checkpoint)
    else:
        model = model_class(config)
    # Load Dataset
    dataset = build_videoae_dataset(args.video_data_path, image_folder=None, sequence_length=args.sequence_length, resolution=config.resolution)
    # Load Trainer
    trainer = VideoAETrainer(model, training_args, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((AEArgument, AETrainingArgument))
    ae_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(ae_args), **vars(training_args))
    set_seed(args.seed)

    train(args, ae_args, training_args)
