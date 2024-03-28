from .configuration_causalvae import CausalVAEConfiguration
from .dataset_causalvae import CausalVAEDataset
from .modeling_causalvae import CausalVAEModel
from .trainer_causalvae import CausalVAETrainer

from einops import rearrange
from torch import nn

class CausalVAEModelWrapper(nn.Module):
    def __init__(self, ckpt):
        super(CausalVAEModelWrapper, self).__init__()
        self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
    def encode(self, x):  # b c t h w
        # x = self.vae.encode(x).sample()
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    def decode(self, x):
        # x = self.vae.decode(x)
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x