from ..configuration_videobase import VideoBaseConfiguration
from typing import Union, Tuple

class CausalVQVAEConfiguration(VideoBaseConfiguration):
    def __init__(
        self,
        embedding_dim: int = 256,
        n_codes: int = 2048,
        n_hiddens: int = 240,
        n_res_layers: int = 4,
        resolution: int = 128,
        sequence_length: int = 16,
        time_downsample: int = 4,
        spatial_downsample: int = 8,
        no_pos_embd: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.n_hiddens = n_hiddens
        self.n_res_layers = n_res_layers
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.time_downsample = time_downsample
        self.spatial_downsample = spatial_downsample
        self.no_pos_embd = no_pos_embd

        self.hidden_size = n_hiddens
