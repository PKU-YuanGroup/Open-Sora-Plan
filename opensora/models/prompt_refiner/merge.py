import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse


def get_lora_model(base_model_path, lora_model_input_path, lora_model_output_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_model_input_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(lora_model_output_path, safe_serialization=True)
    print("Merge lora to base model")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(lora_model_output_path)
    print("Save tokenizer")

def get_model_result(base_model_path, fintune_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    device = "cuda"

    fintune_model = AutoModelForCausalLM.from_pretrained(
        fintune_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    template = "Refine the sentence: \"{}\" to contain subject description, action, scene description. " \
        "(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. " \
        "Make sure it is a fluent sentence, not nonsense."

    prompt = "a dog和一只猫"
    prompt = template.format(prompt)
    messages = [
        {"role": "system", "content": "You are a caption refiner."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    def get_result(model_inputs, model):
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            eos_token_id=tokenizer.get_vocab()["<|eot_id|>"]
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    base_model_response = get_result(model_inputs, base_model)
    fintune_model_response = get_result(model_inputs, fintune_model)
    print("\nInput\n", prompt)
    print("\nResult before fine-tune:\n", base_model_response)
    print("\nResult after fine-tune:\n", fintune_model_response)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="Meta-Llama-3___1-8B-Instruct")
    parser.add_argument("--lora_in_path", type=str, default="llama3_1_instruct_lora/checkpoint-1008")
    parser.add_argument("--lora_out_path", type=str, default="llama3_1_instruct_lora/llama3_8B_lora_merged_cn")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    get_lora_model(args.base_path, args.lora_in_path, args.lora_out_path)
    get_model_result(args.base_path, args.lora_out_path)