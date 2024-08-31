import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from opensora.utils.utils import get_precision


class T5Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(T5Wrapper, self).__init__()
        self.model_name = args.text_encoder_name
        # if 'mt5' in self.model_name:
        from transformers import MT5EncoderModel
        self.text_enc = MT5EncoderModel.from_pretrained("/home/image_data/mt5-xxl", cache_dir=args.cache_dir, **kwargs).eval()
        # self.text_enc = MT5EncoderModel.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl", cache_dir=args.cache_dir, **kwargs).eval()
        # self.text_enc = MT5EncoderModel.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--google--mt5-xl/snapshots/63fc6450d80515b48e026b69ef2fbbd426433e84", cache_dir=args.cache_dir, **kwargs).eval()
        # elif 't5' in self.model_name:
        #     from transformers import T5EncoderModel
        #     # self.text_enc = T5EncoderModel.from_pretrained(self.model_name, cache_dir=args.cache_dir, **kwargs).eval()
        #     self.text_enc = T5EncoderModel.from_pretrained("/storage/ongoing/new/Open-Sora-Plan/cache_dir/models--DeepFloyd--t5-v1_1-xxl/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37", cache_dir=args.cache_dir, **kwargs).eval()

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
    'google/mt5-xl': T5Wrapper,
    'google/mt5-xxl': T5Wrapper,
    'google/umt5-xl': T5Wrapper,
    'google/umt5-xxl': T5Wrapper,
    'DeepFloyd/t5-v1_1-xxl': T5Wrapper,
    'openai/clip-vit-large-patch14': CLIPWrapper
}


def get_text_enc(args):
    """deprecation"""
    encoder_key = None
    for key in text_encoder.keys():
        if key in args.text_encoder_name:
            encoder_key = key
            break
    text_enc = text_encoder.get(encoder_key, None)
    assert text_enc is not None
    return text_enc(args)

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
