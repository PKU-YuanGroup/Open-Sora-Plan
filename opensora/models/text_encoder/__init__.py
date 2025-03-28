from opensora.models.text_encoder.clip import CLIPWrapper
from opensora.models.text_encoder.t5 import T5Wrapper
from opensora.models.text_encoder.glm4 import GLM4Wrapper, ChatGLMModel
from opensora.models.text_encoder.glm4v import GLM4VWrapper, ChatGLMModelV
from opensora.models.text_encoder.qwen2vl import Qwen2VLWrapper
from transformers import AutoTokenizer, ByT5Tokenizer
from opensora.models.text_encoder.glm4_utils import ChatGLM4Tokenizer
from opensora.models.text_encoder.glm4v_utils import ChatGLM4Tokenizer as ChatGLM4TokenizerV
from transformers import Qwen2VLProcessor, T5EncoderModel, MT5EncoderModel, UMT5EncoderModel, \
    CLIPTextModelWithProjection, Qwen2VLModel

text_encoder = {
    'google/mt5-xl': T5Wrapper,
    'google/mt5-xxl': T5Wrapper,
    'google/umt5-xl': T5Wrapper,
    'google/umt5-xxl': T5Wrapper,
    'google/t5-v1_1-xl': T5Wrapper,
    'google/t5-v1_1-xxl': T5Wrapper,
    'clip-vit-large-patch14': CLIPWrapper, 
    'CLIP-ViT-bigG-14-laion2B-39B-b160k': CLIPWrapper, 
    'glm-4-9b-chat': GLM4Wrapper, 
    'glm-4v-9b': GLM4VWrapper, 
    'Qwen2-VL-7B-Instruct': Qwen2VLWrapper, 
    'google/byt5-xxl': T5Wrapper, 
}

text_encoder_cls = {
    'google/mt5-xl': MT5EncoderModel,
    'google/mt5-xxl': MT5EncoderModel,
    'google/umt5-xl': UMT5EncoderModel,
    'google/umt5-xxl': UMT5EncoderModel,
    'google/t5-v1_1-xl': T5EncoderModel,
    'google/t5-v1_1-xxl': T5EncoderModel,
    'clip-vit-large-patch14': CLIPTextModelWithProjection, 
    'CLIP-ViT-bigG-14-laion2B-39B-b160k': CLIPTextModelWithProjection, 
    'glm-4-9b-chat': ChatGLMModel, 
    'glm-4v-9b': ChatGLMModelV, 
    'Qwen2-VL-7B-Instruct': Qwen2VLModel, 
    'google/byt5-xxl': T5EncoderModel, 
}

text_tokenizer = {
    'google/mt5-xl': AutoTokenizer,
    'google/mt5-xxl': AutoTokenizer,
    'google/umt5-xl': AutoTokenizer,
    'google/umt5-xxl': AutoTokenizer,
    'google/t5-v1_1-xl': AutoTokenizer,
    'google/t5-v1_1-xxl': AutoTokenizer,
    'clip-vit-large-patch14': AutoTokenizer, 
    'CLIP-ViT-bigG-14-laion2B-39B-b160k': AutoTokenizer, 
    'glm-4-9b-chat': ChatGLM4Tokenizer, 
    'glm-4v-9b': ChatGLM4TokenizerV, 
    'Qwen2-VL-7B-Instruct': Qwen2VLProcessor, 
    'google/byt5-xxl': ByT5Tokenizer, 
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