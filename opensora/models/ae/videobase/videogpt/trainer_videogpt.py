from torch import nn
from transformers import Trainer
import torch.nn.functional as F

class VideoGPTTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        model = model.module
        x = inputs.get("video")
        z = model.pre_vq_conv(model.encoder(x))
        vq_output = model.codebook(z)
        x_recon = model.decoder(model.post_vq_conv(vq_output['embeddings']))
        recon_loss = F.mse_loss(x_recon, x) / 0.06

        return recon_loss