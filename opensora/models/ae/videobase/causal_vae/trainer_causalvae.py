from ..trainer_videobase import VideoBaseTrainer
from typing import Optional
import os
import torch
from transformers.utils import WEIGHTS_NAME
import json

class CausalVAETrainer(VideoBaseTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        model = model.module
        x = inputs.get("video")
        reconstructions, posterior = model(x)
        aeloss, _ = model.loss(
            x,
            reconstructions,
            posterior,
            split="train",
        )
        return aeloss