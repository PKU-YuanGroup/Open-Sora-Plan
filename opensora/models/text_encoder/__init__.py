import torch
from torch import nn
from transformers import T5EncoderModel, CLIPModel, CLIPProcessor

from opensora.utils.utils import get_precision


class T5Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(T5Wrapper, self).__init__()
        self.model_name = args.text_encoder_name
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, cache_dir=args.cache_dir, **kwargs).eval()

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()

class CLIPWrapper(nn.Module):
    def __init__(self, args):
        super(CLIPWrapper, self).__init__()
        self.model_name = args.text_encoder_name
        dtype = get_precision(args)
        model_kwargs = {'cache_dir': args.cache_dir, 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
        self.text_enc = CLIPModel.from_pretrained(self.model_name, **model_kwargs).eval()

    def forward(self, input_ids, attention_mask): 
        text_encoder_embs = self.text_enc.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return text_encoder_embs.detach()



text_encoder = {
    'DeepFloyd/t5-v1_1-xxl': T5Wrapper,
    'openai/clip-vit-large-patch14': CLIPWrapper
}


def get_text_enc(args):
    """deprecation"""
    text_enc = text_encoder.get(args.text_encoder_name, None)
    assert text_enc is not None
    return text_enc(args)

def get_text_warpper(text_encoder_name):
    """deprecation"""
    text_enc = text_encoder.get(text_encoder_name, None)
    assert text_enc is not None
    return text_enc
