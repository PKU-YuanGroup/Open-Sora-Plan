from ..configuration_videobase import VideoBaseConfiguration
from dataclasses import dataclass
from typing import Tuple

@dataclass
class VideoGPTConfiguration:
    embedding_dim: int
    n_codes: int
    n_hiddens: int
    n_res_layers: int
    resolution: int
    sequence_length: int
    downsample: str
    
    def __post_init__(self):
        if isinstance(self.downsample, str):
            self.downsample = tuple(map(int, self.downsample.split(',')))