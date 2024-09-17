import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM



TEMPLATE = """
Refine the sentence: \"{}\" to contain subject description, action, scene description. " \
"(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. " \
"Make sure it is a fluent sentence, not nonsense.
"""

class OpenSoraCaptionRefiner(nn.Module):
    def __init__(self, args, dtype, device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.caption_refiner, trust_remote_code=True
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.caption_refiner, torch_dtype=dtype, trust_remote_code=True
            ).to(device).eval()
        self.device = device
        
    def get_refiner_output(self, prompt):
        prompt = TEMPLATE.format(prompt)
        messages = [
                {"role": "system", "content": "You are a caption refiner."},
                {"role": "user", "content": prompt}
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response