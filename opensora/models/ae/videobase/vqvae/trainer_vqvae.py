from transformers import Trainer
import torch.nn.functional as F
from typing import Optional
import os
import torch
from transformers.utils import WEIGHTS_NAME
import json

class VQVAETrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        x = inputs.get("video")
        z = model.pre_vq_conv(model.encoder(x))
        vq_output = model.codebook(z)
        x_recon = model.decoder(model.post_vq_conv(vq_output["embeddings"]))
        recon_loss = F.mse_loss(x_recon, x) / 0.06
        commitment_loss = vq_output['commitment_loss']
        loss = recon_loss + commitment_loss
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        with open(os.path.join(output_dir, "config.json"), "w") as file:
            json.dump(self.model.config.to_dict(), file)
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
