from .latte.modeling_latte import Latte_models
from .opensora.modeling_opensora import OpenSora_models
from .udit.modeling_udit import UDiT_models
Diffusion_models = {}
Diffusion_models.update(Latte_models)
Diffusion_models.update(OpenSora_models)
Diffusion_models.update(UDiT_models)

from .latte.modeling_latte import Latte_models_class
from .opensora.modeling_opensora import OpenSora_models_class
from .udit.modeling_udit import UDiT_models_class
Diffusion_models_class = {}
Diffusion_models_class.update(Latte_models_class)
Diffusion_models_class.update(OpenSora_models_class)
Diffusion_models_class.update(UDiT_models_class)
    