from ..configuration_videobase import VideoBaseConfiguration
from dataclasses import dataclass, field
from typing import Tuple, Union

@dataclass
class VQVAEConfiguration:
    embedding_dim: int = field(default=256)
    n_codes: int = field(default=2048)
    n_hiddens: int = field(default=240)
    n_res_layers: int = field(default=4)
    resolution: int = field(default=128)
    sequence_length: int = field(default=16)
    downsample: Union[str, Tuple[int, int, int]] = field(default=(4,4,4))
    no_pos_embd: bool = field(default=True)
    def __post_init__(self):
        if isinstance(self.downsample, str):
            self.downsample = tuple(map(int, self.downsample.split(',')))