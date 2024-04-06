import torch
from diffusers import ModelMixin, ConfigMixin
from torch import nn
import os
import json

"""
We are currently working on the abstraction of the code structure, 
so there is no content here for the moment. 
It will be gradually optimized over time. 
Contributions from the open-source community are welcome.
"""
class VideoBaseAE(ModelMixin, ConfigMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    @classmethod
    def load_from_checkpoint(cls, model_path):
        with open(os.path.join(model_path, "config.json"), "r") as file:
            config = json.load(file)
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model = cls(config=cls.CONFIGURATION_CLS(**config))
        model.load_state_dict(state_dict)
        return model
    
    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        pass
    
    def encode(self, x: torch.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: torch.Tensor, *args, **kwargs):
        pass