"""
new code
"""
from .videogpt import (
    VideoGPTConfiguration,
    VideoGPTVQVAE,
    VideoGPTTrainer,
    VideoGPTDataset,
)


"""
old code
"""
from .vqvae.videogpt import VideoGPTVQVAEWrapper

videovqvae = [
    "bair_stride4x2x2",
    "ucf101_stride4x4x4",
    "kinetics_stride4x4x4",
    "kinetics_stride2x4x4",
]
videovae = []

videobase_ae_stride = {
    'bair_stride4x2x2': [4, 2, 2],
    'ucf101_stride4x4x4': [4, 4, 4],
    'kinetics_stride4x4x4': [4, 4, 4],
    'kinetics_stride2x4x4': [2, 4, 4],
}

videobase_ae_channel = {
    'bair_stride4x2x2': 256,
    'ucf101_stride4x4x4': 256,
    'kinetics_stride4x4x4': 256,
    'kinetics_stride2x4x4': 256,
}

videobase_ae = {
    "bair_stride4x2x2": VideoGPTVQVAEWrapper,
    "ucf101_stride4x4x4": VideoGPTVQVAEWrapper,
    "kinetics_stride4x4x4": VideoGPTVQVAEWrapper,
    "kinetics_stride2x4x4": VideoGPTVQVAEWrapper,
}
