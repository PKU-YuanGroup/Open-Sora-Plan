import os
import json
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.utils import WEIGHTS_NAME


class VideoAETrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        model = model.module
        x = inputs.get("video")
        return model(x)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        with open(os.path.join(output_dir, "config.json"), "w") as file:
            json.dump(self.model.config.to_dict(), file)
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
