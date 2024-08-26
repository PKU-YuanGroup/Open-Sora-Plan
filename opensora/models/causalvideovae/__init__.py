from torchvision.transforms import Lambda
from .model.vae import CausalVAEModel, WFVAEModel
from einops import rearrange
import torch.nn as nn

class CausalVAEModelWrapper(nn.Module):
    def __init__(self, model_path, subfolder=None, cache_dir=None, use_ema=False, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)

    def encode(self, x):
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    def decode(self, x):
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype
    
class WFVAEModelWrapper(nn.Module):
    def __init__(self, model_path, subfolder=None, cache_dir=None, use_ema=False, **kwargs):
        super(WFVAEModelWrapper, self).__init__()
        self.vae = WFVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)

    def encode(self, x):
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    
    def decode(self, x):
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype

ae_wrapper = {
    'CausalVAEModel_D4_2x8x8': CausalVAEModelWrapper,
    'CausalVAEModel_D8_2x8x8': CausalVAEModelWrapper,
    'CausalVAEModel_D4_4x8x8': CausalVAEModelWrapper,
    'CausalVAEModel_D8_4x8x8': CausalVAEModelWrapper,
    'WFVAEModel_D8_4x8x8': WFVAEModelWrapper,
}

ae_stride_config = {
    'CausalVAEModel_D4_2x8x8': [2, 8, 8],
    'CausalVAEModel_D8_2x8x8': [2, 8, 8],
    'CausalVAEModel_D4_4x8x8': [4, 8, 8],
    'CausalVAEModel_D8_4x8x8': [4, 8, 8],
    'WFVAEModel_D8_4x8x8': [4, 8, 8],
}

ae_channel_config = {
    'CausalVAEModel_D4_2x8x8': 4,
    'CausalVAEModel_D8_2x8x8': 8,
    'CausalVAEModel_D4_4x8x8': 4,
    'CausalVAEModel_D8_4x8x8': 8,
    'WFVAEModel_D8_4x8x8': 8,
}

ae_denorm = {
    'CausalVAEModel_D4_2x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_D8_2x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_D4_4x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_D8_4x8x8': lambda x: (x + 1.) / 2.,
    'WFVAEModel_D8_4x8x8': lambda x: (x + 1.) / 2.,
}

ae_norm = {
    'CausalVAEModel_D4_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D4_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'WFVAEModel_D8_4x8x8': Lambda(lambda x: 2. * x - 1.),
}