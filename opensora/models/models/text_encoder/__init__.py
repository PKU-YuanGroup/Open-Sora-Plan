from opensora.models.text_encoder.clip import CLIPWrapper
from opensora.models.text_encoder.t5 import T5Wrapper
from opensora.models.text_encoder.glm4 import GLM4Wrapper
from opensora.models.text_encoder.glm4v import GLM4VWrapper, ChatGLMModel
from opensora.models.text_encoder.qwen2vl import Qwen2VLWrapper
from transformers import AutoTokenizer, ByT5Tokenizer
from opensora.models.text_encoder.glm4_utils import ChatGLM4Tokenizer
from transformers import Qwen2VLProcessor, T5EncoderModel, MT5EncoderModel, UMT5EncoderModel, \
    CLIPTextModelWithProjection, Qwen2VLModel

text_encoder = {
    'mt5-xl': T5Wrapper,
    'mt5-xxl': T5Wrapper,
    'umt5-xl': T5Wrapper,
    'umt5-xxl': T5Wrapper,
    't5-v1_1-xl': T5Wrapper,
    't5-v1_1-xxl': T5Wrapper,
    'clip-vit-large-patch14': CLIPWrapper, 
    'CLIP-ViT-bigG-14-laion2B-39B-b160k': CLIPWrapper, 
    'glm-4-9b-chat': GLM4Wrapper, 
    'glm-4v-9b': GLM4VWrapper, 
    'Qwen2-VL-7B-Instruct': Qwen2VLWrapper, 
    'byt5-xxl': T5Wrapper, 
}

text_encoder_cls = {
    'mt5-xl': MT5EncoderModel,
    'mt5-xxl': MT5EncoderModel,
    'umt5-xl': UMT5EncoderModel,
    'umt5-xxl': UMT5EncoderModel,
    't5-v1_1-xl': T5EncoderModel,
    't5-v1_1-xxl': T5EncoderModel,
    'clip-vit-large-patch14': CLIPTextModelWithProjection, 
    'CLIP-ViT-bigG-14-laion2B-39B-b160k': CLIPTextModelWithProjection, 
    'glm-4-9b-chat': ChatGLMModel, 
    'glm-4v-9b': ChatGLMModel, 
    'Qwen2-VL-7B-Instruct': Qwen2VLModel, 
    'byt5-xxl': T5EncoderModel, 
}

text_tokenizer = {
    'mt5-xl': AutoTokenizer,
    'mt5-xxl': AutoTokenizer,
    'umt5-xl': AutoTokenizer,
    'umt5-xxl': AutoTokenizer,
    't5-v1_1-xl': AutoTokenizer,
    't5-v1_1-xxl': AutoTokenizer,
    'clip-vit-large-patch14': AutoTokenizer, 
    'CLIP-ViT-bigG-14-laion2B-39B-b160k': AutoTokenizer, 
    'glm-4-9b-chat': ChatGLM4Tokenizer, 
    'glm-4v-9b': ChatGLM4Tokenizer, 
    'Qwen2-VL-7B-Instruct': Qwen2VLProcessor, 
    'byt5-xxl': ByT5Tokenizer, 
}

def get_text_warpper(text_encoder_name):
    encoder_key = None
    for key in text_encoder.keys():
        if key in text_encoder_name:
            encoder_key = key
            break
    text_enc = text_encoder.get(encoder_key, None)
    assert text_enc is not None
    return text_enc

def get_text_cls(text_encoder_name):
    encoder_key = None
    for key in text_encoder.keys():
        if key in text_encoder_name:
            encoder_key = key
            break
    text_enc_cls = text_encoder_cls.get(encoder_key, None)
    assert text_enc_cls is not None
    return text_enc_cls

def get_text_tokenizer(text_encoder_name):
    tokenizer_key = None
    for key in text_tokenizer.keys():
        if key in text_encoder_name:
            tokenizer_key = key
            break
    text_tok = text_tokenizer.get(tokenizer_key, None)
    assert text_tok is not None
    return text_tok