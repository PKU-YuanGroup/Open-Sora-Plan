"""
new code
"""
from .vqvae import (
    VQVAEConfiguration,
    VQVAEModel,
    VQVAETrainer,
    VQVAEDataset, VQVAEModelWrapper
)
from .causal_vqvae import (
    CausalVQVAEConfiguration,
    CausalVQVAEDataset,
    CausalVQVAETrainer,
    CausalVQVAEModel, CausalVQVAEModelWrapper
)


"""
old code
"""

videobase_ae_stride = {
    'CausalVQVAEModel': [4, 8, 8],
    'VQVAEModel': [4, 8, 8],
    'bair_stride4x2x2': [4, 2, 2],
    'ucf101_stride4x4x4': [4, 4, 4],
    'kinetics_stride4x4x4': [4, 4, 4],
    'kinetics_stride2x4x4': [2, 4, 4],
}

videobase_ae_channel = {
    'CausalVQVAEModel': 4,
    'VQVAEModel': 4,
    'bair_stride4x2x2': 256,
    'ucf101_stride4x4x4': 256,
    'kinetics_stride4x4x4': 256,
    'kinetics_stride2x4x4': 256,
}

videobase_ae = {
    "CausalVQVAEModel": CausalVQVAEModelWrapper,
    "VQVAEModel": VQVAEModelWrapper,
    "bair_stride4x2x2": VQVAEModelWrapper,
    "ucf101_stride4x4x4": VQVAEModelWrapper,
    "kinetics_stride4x4x4": VQVAEModelWrapper,
    "kinetics_stride2x4x4": VQVAEModelWrapper,
}
