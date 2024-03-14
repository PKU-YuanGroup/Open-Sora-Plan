"""
new code
"""
from .vqvae import (
    VQVAEConfiguration,
    VQVAEModel,
    VQVAETrainer,
    VQVAEDataset, VideoGPTVQVAEWrapper
)


"""
old code
"""

videovqvae = [
    "bair_stride4x2x2",
    "ucf101_stride4x4x4",
    "kinetics_stride4x4x4",
    "kinetics_stride2x4x4",
]
videovae = []

videobase_ae_stride = {
    'checkpoint-14000': [4, 4, 4],
    'bair_stride4x2x2': [4, 2, 2],
    'ucf101_stride4x4x4': [4, 4, 4],
    'kinetics_stride4x4x4': [4, 4, 4],
    'kinetics_stride2x4x4': [2, 4, 4],
}

videobase_ae_channel = {
    'checkpoint-14000': 4,
    'bair_stride4x2x2': 256,
    'ucf101_stride4x4x4': 256,
    'kinetics_stride4x4x4': 256,
    'kinetics_stride2x4x4': 256,
}

videobase_ae = {
    "checkpoint-14000": VideoGPTVQVAEWrapper,
    "bair_stride4x2x2": VideoGPTVQVAEWrapper,
    "ucf101_stride4x4x4": VideoGPTVQVAEWrapper,
    "kinetics_stride4x4x4": VideoGPTVQVAEWrapper,
    "kinetics_stride2x4x4": VideoGPTVQVAEWrapper,
}
