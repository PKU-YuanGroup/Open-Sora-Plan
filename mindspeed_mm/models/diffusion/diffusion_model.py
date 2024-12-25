from torch import nn

from .diffusers_scheduler import DiffusersScheduler
from .opensoraplanv1_5_scheduler import OpenSoraPlanScheduler

DIFFUSION_MODEL_MAPPINGS = {
    "OpenSoraPlan": OpenSoraPlanScheduler
}


class DiffusionModel:
    """
    Factory class for all customized diffusion models and diffusers schedulers.
    Args:
        config:
        {
            "model_id": "ddpm",
            "num_timesteps": 1000,
            "beta_schedule": "linear",
            ...
        }
    """

    def __init__(self, config):
        if config.model_id in DIFFUSION_MODEL_MAPPINGS:
            model_cls = DIFFUSION_MODEL_MAPPINGS[config.model_id]
            self.diffusion = model_cls(**config.to_dict())
        else:
            self.diffusion = DiffusersScheduler(config.to_dict())

    def get_model(self):
        return self.diffusion
