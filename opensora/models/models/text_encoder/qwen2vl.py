import torch
from torch import nn
from transformers import Qwen2VLModel

try:
    import torch_npu
except:
    torch_npu = None

class Qwen2VLWrapper(nn.Module):
    def __init__(self, model_name, **kwargs):
        super(Qwen2VLWrapper, self).__init__()
        self.model_name = model_name
        print(f'Loading Qwen2VL model from {self.model_name}...')
        self.text_enc = Qwen2VLModel.from_pretrained(self.model_name, attn_implementation="flash_attention_2", **kwargs).eval()

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()

if __name__ == '__main__':
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    device = "cuda:0" # the device to load the model onto
    model_path = "/storage/ongoing/12.13/t2i/Open-Sora-Plan/cache_dir/Qwen2-VL-7B-Instruct"
    # model = Qwen2VLForConditionalGeneration.from_pretrained(model_path).to(device)
    # print(type(model))

    processor = Qwen2VLProcessor.from_pretrained(model_path)
    print(type(processor))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me a short introduction to large language model."},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)


    text_enc = Qwen2VLWrapper(model_path).to(device)
    print(text_enc)
    total_params = sum(p.numel() for p in text_enc.parameters())
    print(f"Total parameters: {total_params / 1e9} B")
    model_inputs = processor(
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
    model_inputs_ = processor(text=[text], return_tensors='pt').to(device)
    input_ids_, attention_mask_ = model_inputs_.input_ids, model_inputs_.attention_mask
    logit_ = text_enc(input_ids_, attention_mask_)
    import ipdb;ipdb.set_trace()
    print(torch.allclose(logit[:, :logit_.shape[1], :], logit_))