from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import argparse

def get_output(prompt):
    template = "Refine the sentence: \"{}\" to contain subject description, action, scene description. " \
            "(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. " \
            "Make sure it is a fluent sentence, not nonsense."
    prompt = template.format(prompt)
    messages = [
            {"role": "system", "content": "You are a caption refiner."},
            {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('\nInput\n:', prompt)
    print('\nOutput\n:', response)
    return response

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode_path", type=str, default="llama3_8B_lora_merged_cn")
    parser.add_argument("--prompt", type=str, default='a dog is running.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.mode_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.mode_path,torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

    response = get_output(args.prompt)