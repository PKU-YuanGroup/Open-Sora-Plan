## Data

We have open-sourced our dataset of 32,555 pairs, which includes Chinese data. The dataset is available [here](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/prompt_refiner). The details can be found [here](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md#prompt-refiner).

In fact, it is a JSON file with the following structure.

```
[
  {
    "instruction": "Refine the sentence: \"A newly married couple sharing a piece of there wedding cake.\" to contain subject description, action, scene description. (Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. Make sure it is a fluent sentence, not nonsense.",
    "input": "",
    "output": "The newlywed couple, dressed in elegant attire..."
  },
  ...
]
```

## Train

`--data_path` is the path to the prepared JSON file.
`--model_path` is the directory containing the LLaMA 3.1 weights, including `config.json` and some weight files.
`--lora_out_path` is the path where the LoRA model will be saved.

```
cd opensora/models/prompt_refiner
CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_path path/to/data.json \
    --model_path path/to/llama_model \ 
    --lora_out_path path/to/save/lora_model
```

## Merge

`--model_path` is the directory containing the LLaMA 3.1 weights, including `config.json` and some weight files.
`--lora_in_path` is the directory containing the pre-trained LoRA model.
`--lora_out_path` is the path for the merged model.

```
cd opensora/models/prompt_refiner
CUDA_VISIBLE_DEVICES=0 python merge.py \
    --base_path path/to/llama_model \
    --lora_in_path path/to/save/lora_model \
    --lora_out_path path/to/save/merge_model
```

## Inference

`--model_path` is the directory containing the weights (LLaMA 3.1 or merged Lora weight), including `config.json` and some weight files.
`--prompt` is the text you want to input, which will be refined.

```
cd opensora/models/prompt_refiner
CUDA_VISIBLE_DEVICES=0 python merge.py \
    --mode_path path/to/data.json \
    --prompt path/to/save/lora_model
```