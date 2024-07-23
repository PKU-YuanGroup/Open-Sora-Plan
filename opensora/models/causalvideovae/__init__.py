from torchvision.transforms import Lambda
from .model.causal_vae import CausalVAEModelWrapper

ae_stride_config = {
    'CausalVAEModel_D4_2x8x8': [2, 8, 8],
    'CausalVAEModel_D8_2x8x8': [2, 8, 8],
    'CausalVAEModel_D4_4x8x8': [4, 8, 8],
    'CausalVAEModel_D8_4x8x8': [4, 8, 8],
}


ae_channel_config = {
    'CausalVAEModel_D4_2x8x8': 4,
    'CausalVAEModel_D8_2x8x8': 8,
    'CausalVAEModel_D4_4x8x8': 4,
    'CausalVAEModel_D8_4x8x8': 8,
}


ae_denorm = {
    'CausalVAEModel_D4_2x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_D8_2x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_D4_4x8x8': lambda x: (x + 1.) / 2.,
    'CausalVAEModel_D8_4x8x8': lambda x: (x + 1.) / 2.,
}

ae_norm = {
    'CausalVAEModel_D4_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_2x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D4_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVAEModel_D8_4x8x8': Lambda(lambda x: 2. * x - 1.),
}