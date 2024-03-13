from einops import rearrange
from torch import nn

from .videogpt import load_vqvae


class VideoGPTVQVAEWrapper(nn.Module):
    def __init__(self, ckpt='kinetics_stride4x4x4'):
        super(VideoGPTVQVAEWrapper, self).__init__()
        self.vqvae = load_vqvae(ckpt, root='./cache_dir')
    def encode(self, x):  # b c t h w
        x = self.vqvae.pre_vq_conv(self.vqvae.encoder(x))
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x
    def decode(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w').contiguous()
        vq_output = self.vqvae.codebook(x)
        x = self.vqvae.decoder(self.vqvae.post_vq_conv(vq_output['embeddings']))
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x