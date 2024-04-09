from .vae.vae import HFVAEWrapper
from .vae.vae import SDVAEWrapper
from .vqvae.vqvae import SDVQVAEWrapper

vae = ['stabilityai/sd-vae-ft-mse', 'stabilityai/sd-vae-ft-ema']
vqvae = ['vqgan_imagenet_f16_1024', 'vqgan_imagenet_f16_16384', 'vqgan_gumbel_f8']

imagebase_ae_stride = {
    'stabilityai/sd-vae-ft-mse': [1, 8, 8],
    'stabilityai/sd-vae-ft-ema': [1, 8, 8],
    'vqgan_imagenet_f16_1024': [1, 16, 16],
    'vqgan_imagenet_f16_16384': [1, 16, 16],
    'vqgan_gumbel_f8': [1, 8, 8],
}

imagebase_ae_channel = {
    'stabilityai/sd-vae-ft-mse': 4,
    'stabilityai/sd-vae-ft-ema': 4,
    'vqgan_imagenet_f16_1024': -1,
    'vqgan_imagenet_f16_16384': -1,
    'vqgan_gumbel_f8': -1,
}

imagebase_ae = {
    'stabilityai/sd-vae-ft-mse': HFVAEWrapper,
    'stabilityai/sd-vae-ft-ema': HFVAEWrapper,
    'vqgan_imagenet_f16_1024': SDVQVAEWrapper,
    'vqgan_imagenet_f16_16384': SDVQVAEWrapper,
    'vqgan_gumbel_f8': SDVQVAEWrapper,
}