import pytest
import transformers

from mindspeed_mm import TextEncoder
from mindspeed_mm import Tokenizer
from tests.ut.utils import judge_expression
from tests.ut.utils import TestConfig


T5_MODEL_PATH = "/home/ci_resource/models/t5"
MT5_MODEL_PATH = "/home/ci_resource/models/mt5"
CLIP_MODEL_PATH = "/home/ci_resource/models/stable-diffusion-xl-base-1.0"
T5_TEXT_ENCODER_OUTPUT = -0.50390625
MT5_TEXT_ENCODER_OUTPUT = -1.25
CLIP_TEXT_ENCODER_OUTPUT = -28.099241256713867


class TestTextEncoder:
    """
    text encoder processor test case
    """
    def test_t5(self):
        """
        test t5 text encoder processor
        """
        text_encoder_dict = {
                "hub_backend": "hf",
                "model_id": "T5",
                "dtype": "bf16", 
                "from_pretrained": T5_MODEL_PATH,
        }
        tokenizer_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": T5_MODEL_PATH,
        }
        text_encoder_configs = TestConfig(text_encoder_dict)
        text_encoder = TextEncoder(text_encoder_configs)
        model = text_encoder.get_model()
        judge_expression(isinstance(model, transformers.models.t5.modeling_t5.T5EncoderModel))

        tokenizer_configs = TestConfig(tokenizer_dict)
        tokenizer = Tokenizer(tokenizer_configs)
        t5_tokenizer = tokenizer.get_tokenizer()
        test_text = "This is a T5 example"
        tokenizer_output = t5_tokenizer(test_text, return_tensors='pt')
        output = text_encoder.encode(input_ids=tokenizer_output["input_ids"], mask=tokenizer_output["attention_mask"])
        judge_expression(output["last_hidden_state"].min().item() == T5_TEXT_ENCODER_OUTPUT)

    def test_mt5(self):
        """
        test mt5 text encoder processor
        """
        text_encoder_dict = {
                "hub_backend": "hf",
                "model_id": "MT5",
                "dtype": "bf16",
                "from_pretrained": MT5_MODEL_PATH,
        }
        tokenizer_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": MT5_MODEL_PATH,
        }
        text_encoder_configs = TestConfig(text_encoder_dict)
        text_encoder = TextEncoder(text_encoder_configs)
        model = text_encoder.get_model()
        judge_expression(isinstance(model, transformers.models.mt5.modeling_mt5.MT5EncoderModel))

        tokenizer_configs = TestConfig(tokenizer_dict)
        tokenizer = Tokenizer(tokenizer_configs)
        mt5_tokenizer = tokenizer.get_tokenizer()
        test_text = "This is a MT5 example"
        tokenizer_output = mt5_tokenizer(test_text, return_tensors='pt')
        output = text_encoder.encode(input_ids=tokenizer_output["input_ids"], mask=tokenizer_output["attention_mask"])
        judge_expression(output["last_hidden_state"].min().item() == MT5_TEXT_ENCODER_OUTPUT)

    def test_clip(self):
        """
        test clip text encoder processor
        """
        text_encoder_dict = {
                "hub_backend": "hf",
                "model_id": "CLIP",
                "dtype": "float32",
                "from_pretrained": CLIP_MODEL_PATH,
                "subfolder": "text_encoder"
        }
        tokenizer_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": CLIP_MODEL_PATH,
                "subfolder": "tokenizer"
        }
        text_encoder_configs = TestConfig(text_encoder_dict)
        text_encoder = TextEncoder(text_encoder_configs)
        model = text_encoder.get_model()
        judge_expression(isinstance(model, transformers.models.clip.modeling_clip.CLIPTextModel))

        text_encoder_configs = TestConfig(tokenizer_dict)
        tokenizer = Tokenizer(text_encoder_configs)
        clip_tokenizer = tokenizer.get_tokenizer()
        test_text = "This is a CLIP example"
        tokenizer_output = clip_tokenizer(test_text, return_tensors='pt')
        output = text_encoder.encode(input_ids=tokenizer_output["input_ids"], mask=tokenizer_output["attention_mask"])
        judge_expression(output["last_hidden_state"].min().item() == CLIP_TEXT_ENCODER_OUTPUT)
