import torch
from torch import nn

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
    def load_from_checkpoint(cls, model_path):
        pass
    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        pass
    
    def encode(self, x: torch.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: torch.Tensor, *args, **kwargs):
        pass