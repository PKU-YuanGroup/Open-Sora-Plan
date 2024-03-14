from einops import rearrange
from torch import nn

from .configuration_vqvae import VQVAEConfiguration
from .modeling_vqvae import VQVAEModel
from .trainer_vqvae import VQVAETrainer
from .dataset_vqvae import VQVAEDataset


videovqvae = [
    "bair_stride4x2x2",
    "ucf101_stride4x4x4",
    "kinetics_stride4x4x4",
    "kinetics_stride2x4x4",
]
videovae = []

class VideoGPTVQVAEWrapper(nn.Module):
    def __init__(self, ckpt='kinetics_stride4x4x4'):
        super(VideoGPTVQVAEWrapper, self).__init__()
        if ckpt in videovqvae:
            self.vqvae = VQVAEModel.download_and_load_model(ckpt)
        else:
            self.vqvae = VQVAEModel.load_from_checkpoint(ckpt)
    def encode(self, x):  # b c t h w
        x = self.vqvae.pre_vq_conv(self.vqvae.encoder(x))
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        x = x * 0.18215
        return x
    def decode(self, x):
        x = rearrange(x / 0.18215, 'b t c h w -> b c t h w').contiguous()
        vq_output = self.vqvae.codebook(x)
        x = self.vqvae.decoder(self.vqvae.post_vq_conv(vq_output['embeddings']))
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x