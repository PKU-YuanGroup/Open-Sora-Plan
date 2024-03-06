from .dit.dit import DiT_models
from .latte.latte import Latte_models

Diffusion_models = {}
Diffusion_models.update(DiT_models)
Diffusion_models.update(Latte_models)