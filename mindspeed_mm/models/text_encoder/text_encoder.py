import importlib
import torch
import torch.nn as nn
from mindspeed_mm.utils.utils import get_dtype


TEXT_ENCODER_MAPPING = {
    "T5": "T5EncoderModel",
    "MT5": "MT5EncoderModel",
    "CLIP": "CLIPTextModel",
    "CLIPWithProjection": "CLIPTextModelWithProjection",
}


class TextEncoder(nn.Module):
    """
    Instantiate a text encoder model from config.

    Args:
        config (dict): the general config for Text Encoder Model
        {
            (1) args for our feautrues
            "backend": type-str, "hf" or "om",
            "model_id": type-str, "AutoModel" or other automodel name,
            "dtype": type-str, dtype of text encoder
            
            (2) args for automodel.from_pretrained() of transformers or openmind
            "pretrained_model_name_or_path": type-str, local path or hub path,
            "local_files_only": type-bool,
            ...
        }
    """
    def __init__(self, config, dtype):
        super().__init__()
        config = config.to_dict()
        self.backend = config.pop("hub_backend")
        self.use_attention_mask = config.pop("use_attention_mask", True)
        model_id = config["model_id"]
        if model_id not in TEXT_ENCODER_MAPPING:
            raise ValueError(f"{model_id} text encoder is currently not supported")
        else:
            self.automodel_name = TEXT_ENCODER_MAPPING[config.pop("model_id")]
        config["pretrained_model_name_or_path"] = config.pop("from_pretrained")
        # Only huggingface backend is supported, OpenMind backend will be supported soon.
        module = importlib.import_module("transformers")
        automodel = getattr(module, self.automodel_name)
        self.model = automodel.from_pretrained(**config, torch_dtype=dtype)

    def get_model(self):
        return self.model

    def encode(self, input_ids, mask, **kwargs):
        attention_mask = mask if self.use_attention_mask else None
        if self.automodel_name != "CLIPTextModelWithProjection":
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )['last_hidden_state']
        else:
            output = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )[0]
        return output
