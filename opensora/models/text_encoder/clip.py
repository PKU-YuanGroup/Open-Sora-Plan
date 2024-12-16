import torch
from torch import nn
from transformers import CLIPTextModelWithProjection

try:
    import torch_npu
except:
    torch_npu = None

class CLIPWrapper(nn.Module):
    def __init__(self, model_name, **kwargs):
        super(CLIPWrapper, self).__init__()
        self.model_name = model_name
        print(f'Loading CLIP model from {self.model_name}...')
        self.text_enc = CLIPTextModelWithProjection.from_pretrained(self.model_name, **kwargs).eval()

    def forward(self, input_ids, attention_mask): 
        text_encoder_embs = self.text_enc(input_ids=input_ids, output_hidden_states=True)[0]
        return text_encoder_embs.detach()
