from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from torch import nn

import mindspeed_mm
from mindspeed_mm.models.sora_model import SoRAModel
from tests.ut.utils import judge_expression


def dict_to_obj(data: Dict[str, Any]) -> Any:
    """convert a dictionary to an object recursively"""
    if isinstance(data, dict):
        data_dict = {k: dict_to_obj(v) for k, v in data.items()}
        return SimpleNamespace(**data_dict)
    if isinstance(data, list):
        data_list = [dict_to_obj(item) for item in data]
        return data_list
    return data


class AEDummyObject:
    pass


class TextEncoderDummyObject:
    pass


class PredictModelDummyObject:
    pass


class DiffusionModelDummyObject:
    pass


class AEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_model(self):
        return AEDummyObject()


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_model(self):
        return TextEncoderDummyObject()


class PredictModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_model(self):
        return PredictModelDummyObject()


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_model(self):
        return DiffusionModelDummyObject()


mm_config = {
    "ae":{
        "model_id":"video_gpt",
        "channels":3,
        "hidden_size":2,
        "dropout":0.0,
    },
    "text_encoder":{
        "model_id":"t5",
        "backend":"hf",
        "from_pretrained":"mt5/xxl"
    },
    "predictor":{
        "model_id":"dit",
        "hidden_size":5
    },
    "diffusion":{
        "model_id":"iddpm",
        "timesteps":1000
    },
    "load_video_features":False,
    "load_text_features":False,
    "device":"npu",
    "dtype":"fp32"
}


@pytest.fixture
def config():
    return mm_config


class TestSoRAModel:
    def test_model_build(self, config):
        # patchs
        mindspeed_mm.models.sora_model.AEModel = AEModel
        mindspeed_mm.models.sora_model.TextEncoder = TextEncoder
        mindspeed_mm.models.sora_model.PredictModel = PredictModel
        mindspeed_mm.models.sora_model.DiffusionModel = DiffusionModel

        config = dict_to_obj(config)
        sora = SoRAModel(config)
        judge_expression(isinstance(sora.ae, AEDummyObject))
        judge_expression(isinstance(sora.text_encoder, TextEncoderDummyObject))
        judge_expression(isinstance(sora.predictor, PredictModelDummyObject))
        judge_expression(isinstance(sora.diffusion, DiffusionModelDummyObject))
