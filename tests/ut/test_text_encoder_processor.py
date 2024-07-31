import pytest
import transformers

from mindspeed_mm import TextEncoder
from tests.ut.utils import judge_expression


T5_MODEL_PATH = "/home/ci_resource/models/t5"
MT5_MODEL_PATH = "/home/ci_resource/models/mt5"
CLIP_MODEL_PATH = "/home/ci_resource/models/stable-diffusion-xl-base-1.0"


class TestTextEncoder:
    """
    text encoder processor test case
    """
    def test_t5(self):
        """
        test t5 text encoder processor
        """
        configs = {
                "hub_backend": "hf",
                "automodel_name": "AutoModel",
                "pretrained_model_name_or_path": T5_MODEL_PATH,
                "local_files_only": False
        }
        text_encoder = TextEncoder(configs)
        model = text_encoder.get_model()
        judge_expression(isinstance(model, transformers.models.t5.modeling_t5.T5Model))
        
    def test_mt5(self):
        """
        test mt5 text encoder processor
        """
        configs = {
                "hub_backend": "hf",
                "automodel_name": "AutoModel",
                "pretrained_model_name_or_path": MT5_MODEL_PATH,
                "local_files_only": False
        }
        text_encoder = TextEncoder(configs)
        model = text_encoder.get_model()
        judge_expression(isinstance(model, transformers.models.mt5.modeling_mt5.MT5Model))

    def test_clip(self):
        """
        test clip text encoder processor
        """
        configs = {
                "hub_backend": "hf",
                "automodel_name": "CLIPTextModel",
                "pretrained_model_name_or_path": CLIP_MODEL_PATH,
                "local_files_only": False,
                "subfolder": "text_encoder"
        }
        text_encoder = TextEncoder(configs)
        model = text_encoder.get_model()
        judge_expression(isinstance(model, transformers.models.clip.modeling_clip.CLIPTextModel))
