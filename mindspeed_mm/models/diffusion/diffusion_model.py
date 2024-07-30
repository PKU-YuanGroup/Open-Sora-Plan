from torch import nn

from .ddpm import DDPM
from .iddpm import IDDPM
from .rflow import RFlow


DIFFUSION_MODEL_MAPPINGS = {
    "ddpm": DDPM,
    "iddpm": IDDPM,
    "rflow": RFlow
}


class DiffusionModel(nn.Module):
    def __init__(self, config):
        model_cls = DIFFUSION_MODEL_MAPPINGS.get(config.model_id)
        self.diffusion = model_cls(config)

    def get_model(self):
        return self.diffusion
