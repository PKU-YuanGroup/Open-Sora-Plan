from torch import nn
from megatron.training.utils import print_rank_0

from mindspeed_mm.models.common.checkpoint import load_checkpoint
from .dits import OpenSoraT2V_v1_3, OpenSoraInpaint_v1_3, VideoDiT, Latte, STDiT, STDiT3, CogVideoX

PREDICTOR_MODEL_MAPPINGS = {
    "OpenSoraT2V_v1_3": OpenSoraT2V_v1_3,
    "OpenSoraI2V_v1_3": OpenSoraInpaint_v1_3,
    "videodit": VideoDiT,
    "latte": Latte,
    "stdit": STDiT,
    "stdit3": STDiT3,
    "cogvideox": CogVideoX
}


class PredictModel(nn.Module):
    """
    The backnone of the denoising model
    PredictModel is the factory class for all unets and dits

    Args:
        config[dict]: for Instantiating an atomic methods
    """

    def __init__(self, config):
        super().__init__()
        model_cls = PREDICTOR_MODEL_MAPPINGS[config.model_id]
        self.predictor = model_cls(**config.to_dict())
        if config.from_pretrained is not None and config.model_id != "cogvideox":
            load_checkpoint(self.predictor, config.from_pretrained)
            print_rank_0("load predictor's checkpoint sucessfully")

    def get_model(self):
        return self.predictor
