from .latte.modeling_latte import Latte_models
from .opensora.modeling_opensora import OpenSora_models
Diffusion_models = {}
Diffusion_models.update(Latte_models)
Diffusion_models.update(OpenSora_models)

    