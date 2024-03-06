
from torch import nn
from diffusers.models import AutoencoderKL


class HFVAEWrapper(nn.Module):
    def __init__(self, hfvae='mse'):
        super(HFVAEWrapper, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{hfvae}', cache_dir='cache_dir')
    def encode(self, x):  # b c h w
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        return x
    def decode(self, x):
        x = self.vae.decode(x / 0.18215).sample
        return x

class SDVAEWrapper(nn.Module):
    def __init__(self):
        super(SDVAEWrapper, self).__init__()
        raise NotImplementedError

    def encode(self, x):  # b c h w
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError