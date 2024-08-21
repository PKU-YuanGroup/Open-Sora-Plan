import pytest
import transformers

from mindspeed_mm import Tokenizer
from tests.ut.utils import judge_expression
from tests.ut.utils import TestConfig


T5_MODEL_PATH = "/home/ci_resource/models/t5"
MT5_MODEL_PATH = "/home/ci_resource/models/mt5"
CLIP_MODEL_PATH = "/home/ci_resource/models/stable-diffusion-xl-base-1.0"
T5_TOKENIZER_OUTPUT = {"input_ids": [100, 19, 3, 9, 332, 755, 677, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
MT5_TOKENIZER_OUTPUT = {"input_ids": [1494, 339, 259, 262, 15576, 428, 11310, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
CLIP_TOKENIZER_OUTPUT = {"input_ids": [49406, 589, 533, 320, 9289, 6228, 49407], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}


class TestTokenizer:
    """
    text tokenizer test case
    """
    def test_t5(self):
        """
        test t5 tokenizer processor
        """
        config_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": T5_MODEL_PATH,
        }
        configs = TestConfig(config_dict)
        tokenizer = Tokenizer(configs)
        t5_tokenizer = tokenizer.get_tokenizer()
        judge_expression(isinstance(t5_tokenizer, transformers.models.t5.tokenization_t5_fast.T5TokenizerFast))

        test_text = "This is a T5 example"
        output = t5_tokenizer(test_text)
        judge_expression(output == T5_TOKENIZER_OUTPUT)
        
    def test_mt5(self):
        """
        test mt5 tokenizer processor
        """
        config_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": MT5_MODEL_PATH,
        }
        configs = TestConfig(config_dict)
        tokenizer = Tokenizer(configs)
        mt5_tokenizer = tokenizer.get_tokenizer()
        judge_expression(isinstance(mt5_tokenizer, transformers.models.t5.tokenization_t5_fast.T5TokenizerFast))

        test_text = "This is a MT5 example"
        output = mt5_tokenizer(test_text)
        judge_expression(output == MT5_TOKENIZER_OUTPUT)

    def test_clip(self):
        """
        test clip tokenizer processor
        """
        config_dict = {
                "hub_backend": "hf",
                "autotokenizer_name": "AutoTokenizer",
                "from_pretrained": CLIP_MODEL_PATH,
                "subfolder": "tokenizer"
        }
        configs = TestConfig(config_dict)
        tokenizer = Tokenizer(configs)
        clip_tokenizer = tokenizer.get_tokenizer()
        judge_expression(isinstance(clip_tokenizer, transformers.models.clip.tokenization_clip_fast.CLIPTokenizerFast))

        test_text = "This is a CLIP example"
        output = clip_tokenizer(test_text)
        judge_expression(output == CLIP_TOKENIZER_OUTPUT)
