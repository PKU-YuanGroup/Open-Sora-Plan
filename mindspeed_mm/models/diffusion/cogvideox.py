import importlib
import torch.nn as nn

COGVIDEOX_SCHEDULER_MAPPING = {
    "cogvideox_2b": "CogVideoXDDIMScheduler",
    "cogvideox_5b": "CogVideoXDPMScheduler",
}


class CogVideoXScheduler(nn.Module):
    def __init__(
        self,
        model_version: str = "cogvideox_5b",
        from_pretrained: str = "",
        **kwargs
    ):
        super().__init__()
        self.automodel_name = COGVIDEOX_SCHEDULER_MAPPING[model_version]

        config = {"pretrained_model_name_or_path": from_pretrained}
        module = importlib.import_module("diffusers")
        automodel = getattr(module, self.automodel_name)
        self.model = automodel.from_pretrained(**config)

