import torch
from torch import nn
from transformers import T5EncoderModel

try:
    import torch_npu
except:
    torch_npu = None

class T5Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(T5Wrapper, self).__init__()
        # self.model_name = args.text_encoder_name_1
        self.model_name = '/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl'
        # self.model_name = '/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/byt5-xxl'
        # self.model_name = '/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl'
        print(f'Loading T5 model from {self.model_name}...')
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, cache_dir=args.cache_dir, **kwargs).eval()

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()


if __name__ == '__main__':
    from transformers import T5Tokenizer, T5ForConditionalGeneration, ByT5Tokenizer
    device = "cuda:0" # the device to load the model onto

    tokenizer = T5Tokenizer.from_pretrained("/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl")
    # tokenizer = ByT5Tokenizer.from_pretrained("/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/byt5-xxl")
    model_inputs = tokenizer("Give me a short introduction to large language model.", return_tensors="pt")

    # model = T5ForConditionalGeneration.from_pretrained("/storage/cache_dir/t5-v1_1-xl")
    # outputs = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    args = type('args', (), 
        {
            'cache_dir': 'a', 
        }
    )
    text_enc = T5Wrapper(args).to(device)
    print(text_enc)
    total_params = sum(p.numel() for p in text_enc.parameters())
    print(f"Total parameters: {total_params / 1e9} B")
    input_ids, attention_mask = model_inputs.input_ids.to(device), model_inputs.attention_mask.to(device)
    logit = text_enc(input_ids, attention_mask)
    print(logit.shape)
    for i in range(logit.shape[1]):
        print(logit[:, i].max().item(), logit[:, i].min().item(), logit[:, i].mean().item(), logit[:, i].std().item())