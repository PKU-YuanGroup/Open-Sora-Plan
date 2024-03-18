import torch
from torch import nn
from transformers import T5EncoderModel, CLIPModel, CLIPProcessor

from opensora.utils.utils import get_precision


class T5Wrapper(nn.Module):
    def __init__(self, args):
        super(T5Wrapper, self).__init__()
        self.model_name = args.text_encoder_name
        dtype = get_precision(args)
        t5_model_kwargs = {'cache_dir': './cache_dir', 'low_cpu_mem_usage': True, 'torch_dtype': dtype}
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, **t5_model_kwargs).eval()

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()

class CLIPWrapper(nn.Module):
    def __init__(self, args, device='cuda', cache_dir='./cache_dir', **kwargs):
        super(CLIPWrapper, self).__init__()
        self.device = torch.device(device)
        self.model_name = args.clip_model_name
        self.cache_dir = cache_dir
        
        # Load the CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def forward(self, prompts): # prompts = ['I am a test caption', 'Test twice']
        # Process the prompts and move to the correct device
        inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Generate text features
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        
        return text_features.detach()



text_encoder = {
    'DeepFloyd/t5-v1_1-xxl': T5Wrapper,
    'clip': CLIPWrapper
}


def get_text_enc(args):
    """deprecation"""
    text_enc = text_encoder.get(args.text_encoder_name, None)
    assert text_enc is not None
    return text_enc(args)
