import torch
from torch import nn
from transformers import CLIPTextModelWithProjection


class CLIPWrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(CLIPWrapper, self).__init__()
        self.model_name = args.text_encoder_name_2
        self.model_name = '/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        self.text_enc = CLIPTextModelWithProjection.from_pretrained(self.model_name, cache_dir=args.cache_dir, **kwargs).eval()

    def forward(self, input_ids, attention_mask): 
        text_encoder_embs = self.text_enc(input_ids=input_ids, output_hidden_states=True)[0]
        return text_encoder_embs.detach()
