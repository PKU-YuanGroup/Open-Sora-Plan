from .vqvae.videogpt import VideoGPTVQVAEWrapper

videovqvae = ['bair_stride4x2x2', 'ucf101_stride4x4x4', 'kinetics_stride4x4x4', 'kinetics_stride2x4x4']
videovae = []

videobase_ae = {
    'bair_stride4x2x2': VideoGPTVQVAEWrapper,
    'ucf101_stride4x4x4': VideoGPTVQVAEWrapper,
    'kinetics_stride4x4x4': VideoGPTVQVAEWrapper,
    'kinetics_stride2x4x4': VideoGPTVQVAEWrapper,
}