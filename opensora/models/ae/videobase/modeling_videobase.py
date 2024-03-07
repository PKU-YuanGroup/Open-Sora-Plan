from torch import nn
from transformers import PreTrainedModel
import os
from typing import Union

"""
We are currently working on the abstraction of the code structure, 
so there is no content here for the moment. 
It will be gradually optimized over time. 
Contributions from the open-source community are welcome.
"""


class VideoBaseAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: str = None,
    ):
        """Automatically determine whether to load from checkpoint or download from Internet"""
        if os.path.exists(pretrained_model_name_or_path):
            return cls.load_from_checkpoint(pretrained_model_name_or_path)
        else:
            return cls.download_and_load_model(
                pretrained_model_name_or_path, cache_dir=cache_dir
            )

    @classmethod
    def load_from_checkpoint(cls, model_path: os.PathLike):
        pass

    @classmethod
    def download_and_load_model(cls, model_name: str, cache_dir=None):
        pass

    def encode(self, x, *args, **kwargs):
        pass

    def decode(self, encoding, *args, **kwargs):
        pass
