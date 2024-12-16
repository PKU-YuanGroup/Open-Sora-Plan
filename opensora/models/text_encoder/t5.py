import torch
from torch import nn
from transformers import T5EncoderModel

try:
    import torch_npu
except:
    torch_npu = None

class T5Wrapper(nn.Module):
    def __init__(self, model_name, **kwargs):
        super(T5Wrapper, self).__init__()
        self.model_name = model_name
        print(f'Loading T5 model from {self.model_name}...')
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, **kwargs).eval()

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()


if __name__ == '__main__':
    from transformers import T5Tokenizer, T5ForConditionalGeneration, ByT5Tokenizer
    device = "cuda:0" # the device to load the model onto
    model_path = "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/google/t5-v1_1-xxl"
    text = "Give me a short introduction to large language model."
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model_inputs = tokenizer(text=[text], return_tensors="pt")

    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    # outputs = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    text_enc = T5Wrapper(model_path).to(device)
    print(text_enc)
    total_params = sum(p.numel() for p in text_enc.parameters())
    print(f"Total parameters: {total_params / 1e9} B")
    model_inputs = tokenizer(
        text=[text], 
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
        ).to(device)
    input_ids, attention_mask = model_inputs.input_ids, model_inputs.attention_mask
    logit = text_enc(input_ids, attention_mask)
    # print(logit.shape)
    # for i in range(logit.shape[1]):
    #     print(logit[:, i].max().item(), logit[:, i].min().item(), logit[:, i].mean().item(), logit[:, i].std().item())
    model_inputs_ = tokenizer(text=[text], return_tensors='pt').to(device)
    input_ids_, attention_mask_ = model_inputs_.input_ids, model_inputs_.attention_mask
    logit_ = text_enc(input_ids_, attention_mask_)
    import ipdb;ipdb.set_trace()
    print(torch.allclose(logit[:, :logit_.shape[1], :], logit_))