import sys
sys.path.append(".")
from opensora.models.ae.videobase import (
    VideoGPTVQVAE,
    VideoGPTConfiguration,
    VideoGPTDataset,
    VideoGPTTrainer,
)
import argparse
import accelerate
from typing import Optional
from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field


@dataclass
class VideoGPTArgument(VideoGPTConfiguration):
    data_path: str = field(default=None, metadata={"help": "data path"})

@dataclass
class VideoGPTTrainingArgument(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

def train(args, videogpt_args, training_args):
    # Load Model
    model_config = VideoGPTConfiguration(
        embedding_dim=args.embedding_dim,
        n_codes=args.n_codes,
        n_hiddens=args.n_hiddens,
        n_res_layers=args.n_res_layers,
        sequence_length=args.sequence_length,
        downsample=args.downsample,
        resolution=args.resolution
    )
    model = VideoGPTVQVAE(model_config)
    # Load Dataset
    dataset = VideoGPTDataset(args.data_path, sequence_length=args.sequence_length)

    # Load Trainer
    trainer = VideoGPTTrainer(model, training_args, train_dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((VideoGPTArgument, VideoGPTTrainingArgument))
    videogpt_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(videogpt_args), **vars(training_args))
    set_seed(args.seed)

    train(args, videogpt_args, training_args)
