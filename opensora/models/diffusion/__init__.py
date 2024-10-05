
from .opensora_v1_2.modeling_opensora import OpenSora_v1_2_models
from .opensora_v1_3.modeling_opensora import OpenSora_v1_3_models
from .opensora_v1_5.modeling_opensora import OpenSora_v1_5_models

Diffusion_models = {}
Diffusion_models.update(OpenSora_v1_2_models)
Diffusion_models.update(OpenSora_v1_3_models)
Diffusion_models.update(OpenSora_v1_5_models)

from .opensora_v1_2.modeling_opensora import OpenSora_v1_2_models_class
from .opensora_v1_3.modeling_opensora import OpenSora_v1_3_models_class
from .opensora_v1_5.modeling_opensora import OpenSora_v1_5_models_class

Diffusion_models_class = {}
Diffusion_models_class.update(OpenSora_v1_2_models_class)
Diffusion_models_class.update(OpenSora_v1_3_models_class)
Diffusion_models_class.update(OpenSora_v1_5_models_class)
    