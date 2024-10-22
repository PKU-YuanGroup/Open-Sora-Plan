from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
import argparse

ins = "Refine the sentence to contain subject description, action, scene description. " \
        "(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. " \
        "Make sure it is a fluent sentence, not nonsense."

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='refine_32255.json')
    parser.add_argument("--model_path", type=str, default='Meta-Llama-3___1-8B-Instruct')
    parser.add_argument("--lora_out_path", type=str, default="llama3_1_instruct_lora")
    args = parser.parse_args()
    return args

args = parse_args()


df = pd.read_json(args.data_path)
ds = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
    MAX_LENGTH = 2048   
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a caption refiner.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] 
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH: 
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto",torch_dtype=torch.bfloat16)
print(model)
model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=64,
    lora_dropout=0.1
)
print(config)

model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=args.lora_out_path,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    logging_steps=1,
    num_train_epochs=1,
    save_steps=20, 
    dataloader_num_workers=4, 
    learning_rate=1.5e-4,
    warmup_ratio=0.1, 
    save_on_each_node=True,
    gradient_checkpointing=True, 
    report_to='wandb', 
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()