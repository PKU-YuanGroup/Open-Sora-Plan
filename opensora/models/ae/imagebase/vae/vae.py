from einops import rearrange
from torch import nn
from diffusers.models import AutoencoderKL


class HFVAEWrapper(nn.Module):
    def __init__(self, hfvae='mse'):
        super(HFVAEWrapper, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(hfvae, cache_dir='cache_dir')
    def encode(self, x):  # b c h w
        t = 0
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        if t != 0:
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t).contiguous()
        return x
    def decode(self, x):
        t = 0
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()
        x = self.vae.decode(x / 0.18215).sample
        if t != 0:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=t).contiguous()
        return x

class SDVAEWrapper(nn.Module):
    def __init__(self):
        super(SDVAEWrapper, self).__init__()
        raise NotImplementedError

    def encode(self, x):  # b c h w
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError