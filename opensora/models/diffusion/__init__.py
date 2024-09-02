
from .opensora.modeling_opensora import OpenSora_models
from .opensora1.modeling_opensora import OpenSora1_models
from .opensora2.modeling_opensora import OpenSora2_models
from .udit.modeling_udit import UDiT_models

from .opensora2.modeling_inpaint import OpenSoraInpaint_models

Diffusion_models = {}
Diffusion_models.update(OpenSora_models)
Diffusion_models.update(OpenSora1_models)
Diffusion_models.update(OpenSora2_models)
Diffusion_models.update(UDiT_models)
Diffusion_models.update(OpenSoraInpaint_models)

from .opensora.modeling_opensora import OpenSora_models_class
from .opensora1.modeling_opensora import OpenSora1_models_class
from .opensora2.modeling_opensora import OpenSora2_models_class
from .udit.modeling_udit import UDiT_models_class

from .opensora2.modeling_inpaint import OpenSoraInpaint_models_class

Diffusion_models_class = {}
Diffusion_models_class.update(OpenSora_models_class)
Diffusion_models_class.update(OpenSora1_models_class)
Diffusion_models_class.update(OpenSora2_models_class)
Diffusion_models_class.update(UDiT_models_class)

Diffusion_models_class.update(OpenSoraInpaint_models_class)
    