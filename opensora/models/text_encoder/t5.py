import torch
from torch import nn
from transformers import T5EncoderModel, UMT5EncoderModel

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
    from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, ByT5Tokenizer
    device = "cuda:0" # the device to load the model onto
    # model_path = "/storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl"
    model_path = "/storage/ongoing/12.13/t2i/cache_dir/google/byt5-xxl"
    # model_path = "/storage/ongoing/new/Open-Sora-Plan/cache_dir/mt5-xxl"
    # model_path = "/storage/ongoing/12.13/t2i/cache_dir/google/umt5-xxl"
    text = "Give me a short introduction to large language model."
    # text = "The image portrays a scene of domestic life, captured in the style of 17th century Dutch genre painting. A woman, dressed in a blue dress and a white headscarf, is seated at a table in a kitchen. She appears to be in a state of exhaustion, her head resting on her hands. The table is laden with various objects, including a red bowl, a green bottle, and a white pitcher. A wooden chair and a broom are also present in the room, adding to the homely atmosphere. The painting is executed in oil on canvas, a common medium for genre paintings of this period. The artist has skillfully used this medium to create a realistic and detailed depiction of everyday life."
    # text = "The image presents a striking digital illustration of a mythical creature that seems to be a fusion of a fish and a human. The creature's head is that of a fish, complete with a mouth full of sharp teeth and a pair of horns on its head. Its body, however, is human-like, with a muscular torso and arms. The creature is set against the backdrop of a dark cave, with a single light source illuminating it from above. The color palette of the image is dominated by shades of blue and green, adding to the overall mysterious and otherworldly atmosphere of the scene. The creature's position in the center of the image draws the viewer's attention immediately, making it the focal point of the composition. The image does not contain any discernible text. The relative position of the creature to the cave and the light source suggests that the creature is emerging from the depths of the cave, further enhancing the sense of mystery and intrigue. The image does not contain any other objects or creatures, making the mythical fish-human hybrid the sole focus of the viewer's attention. The image does not provide any information about the actions of the objects, as the creature appears to be stationary. The image does not contain any aesthetic descriptions. The image does not contain any imaginary content; all descriptions are based on the visible content of the image. The image does not contain any content that can be confidently determined as being outside the image. The image does not contain any aesthetic descriptions. The image does not contain any imaginary content; all descriptions are based on the visible content of the image. The image does not contain any content that can be confidently determined as being outside the image. The image does not contain any aesthetic descriptions. The image does not contain any imaginary content; all descriptions are based on the visible content of the image. The image does not contain any content that can be confidently determined as being outside the image. The image does not contain any aesthetic descriptions. The image does not contain any imaginary content; all descriptions are based on the visible content of the image. The image does not contain any content that can be confidently determined as being outside the image. The image does not contain any aesthetic descriptions. The image does not contain any imaginary content; all descriptions are based on the visible content of the image. The image does not contain any content that can be confidently determined as being outside the image. The image does not contain any aesthetic descriptions. The image does not contain any imaginary content; all descriptions are based on the"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_inputs = tokenizer(text=[text], return_tensors="pt")
    model_path = "/storage/ongoing/12.13/t2i/cache_dir/google/t5-v1_1-xxl"
    tokenizer_ = T5Tokenizer.from_pretrained(model_path)
    model_inputs_ = tokenizer_(text=[text], return_tensors="pt")
    import ipdb;ipdb.set_trace()
    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    # outputs = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    text_enc = T5Wrapper(model_path).to(device)
    # print(text_enc)
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


    text_enc_ = T5Wrapper(model_path, umt5=True).to(device)
    print(text_enc_)
    total_params = sum(p.numel() for p in text_enc_.parameters())
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
    logit_ = text_enc(input_ids, attention_mask)
    import ipdb;ipdb.set_trace()
    # print(logit.shape)
    # for i in range(logit.shape[1]):
    #     print(logit[:, i].max().item(), logit[:, i].min().item(), logit[:, i].mean().item(), logit[:, i].std().item())
    model_inputs_ = tokenizer(text=[text], return_tensors='pt').to(device)
    input_ids_, attention_mask_ = model_inputs_.input_ids, model_inputs_.attention_mask
    logit_ = text_enc(input_ids_, attention_mask_)
    import ipdb;ipdb.set_trace()
    print(torch.allclose(logit[:, :logit_.shape[1], :], logit_))