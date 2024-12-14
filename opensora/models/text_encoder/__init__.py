from opensora.models.text_encoder.clip import CLIPWrapper
from opensora.models.text_encoder.t5 import T5Wrapper
from opensora.models.text_encoder.glm4 import GLM4Wrapper
from opensora.models.text_encoder.glm4v import GLM4VWrapper
from opensora.models.text_encoder.qwen2vl import Qwen2VLWrapper

text_encoder = {
    'google/mt5-xl': T5Wrapper,
    'google/mt5-xxl': T5Wrapper,
    'google/umt5-xl': T5Wrapper,
    'google/umt5-xxl': T5Wrapper,
    'google/t5-v1_1-xl': T5Wrapper,
    'google/t5-v1_1-xxl': T5Wrapper,
    'DeepFloyd/t5-v1_1-xxl': T5Wrapper,
    'openai/clip-vit-large-patch14': CLIPWrapper, 
    'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k': CLIPWrapper, 
    'THUDM/glm-4-9b-chat': GLM4Wrapper, 
    'THUDM/glm-4v-9b': GLM4VWrapper, 
    'Qwen/Qwen2-VL-7B-Instruct': Qwen2VLWrapper, 
    'google/byt5-xxl': T5Wrapper, 
}

def get_text_warpper(text_encoder_name):
    """deprecation"""
    encoder_key = None
    for key in text_encoder.keys():
        if key in text_encoder_name:
            encoder_key = key
            break
    text_enc = text_encoder.get(encoder_key, None)
    assert text_enc is not None
    return text_enc
