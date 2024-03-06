import sys
sys.path.append(".")
from opensora.models.ae.videobase import (
    VQVAEModel,
    VQVAEConfiguration,
    VQVAEDataset,
    VQVAETrainer,
)
import argparse
from typing import Optional
from accelerate.utils import set_seed
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field


@dataclass
class VQVAEArgument(VQVAEConfiguration):
    data_path: str = field(default=None, metadata={"help": "data path"})

@dataclass
class VQVAETrainingArgument(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

def train(args, vqvae_args, training_args):
    # Load Model
    model = VQVAEModel(vqvae_args)
    # Load Dataset
    dataset = VQVAEDataset(args.data_path, sequence_length=args.sequence_length)

    # Load Trainer
    trainer = VQVAETrainer(model, training_args, train_dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((VQVAEArgument, VQVAETrainingArgument))
    vqvae_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(vqvae_args), **vars(training_args))
    set_seed(args.seed)

    train(args, vqvae_args, training_args)
