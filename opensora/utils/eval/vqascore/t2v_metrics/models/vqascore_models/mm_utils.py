from PIL import Image
import os

import torch
from transformers import AutoTokenizer
from ...constants import HF_CACHE_DIR, IMAGE_TOKEN_INDEX



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def t5_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    # Since there's no bos_token_id, simply concatenate the tokenized prompt_chunks with the image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def load_pretrained_model(model_cls,
                          model_args,
                          model_path=None,
                          tokenizer_path=None,
                          model_max_length=None,
                          padding_side=None,
                          image_aspect_ratio='pad', # or 'square'
                          mmprojector_repo=None,
                          mmprojector_name=None,
                          device='cuda',
                          cache_dir=HF_CACHE_DIR):
    tokenizer_dict = {}
    if model_max_length:
        tokenizer_dict['model_max_length'] = model_max_length
    if padding_side:
        tokenizer_dict['padding_side'] = padding_side
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, **tokenizer_dict)
    # tokenizer.pad_token = tokenizer.unk_token # could be redundant
    model = model_cls.from_pretrained(model_path, cache_dir=cache_dir)
    
    if mmprojector_repo:
        from huggingface_hub import hf_hub_download
        model_base_name = mmprojector_repo.split('/')[-1]
        
        if cache_dir is not None:
            local_dir = os.path.join(cache_dir, model_base_name)
        elif os.environ.get('HF_HOME') is not None:
            local_dir = os.path.join(os.environ.get('HF_HOME'), model_base_name)
        else:
            local_dir = os.path.join(os.path.expanduser("~"), model_base_name)
        print(f"Downloading projector weights to {local_dir}")
        hf_hub_download(
            repo_id=mmprojector_repo,
            filename=mmprojector_name,
            local_dir=local_dir,
        )
        pretrain_mm_mlp_adapter = os.path.join(local_dir, mmprojector_name)
        model_args.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter # important to set to correct path
        
        model.get_model().initialize_vision_modules(model_args) # This will load the CLIP vision encoder and MLP projector
    else:
        model.resize_token_embeddings(len(tokenizer)) # perhaps not needed

    if not model.get_vision_tower().is_loaded:
        model.get_vision_tower().load_model()
    model.to(device=device, dtype=torch.bfloat16)
    image_processor = model.get_vision_tower().image_processor

    model.requires_grad_(False)
    
    
    # below might be redundant
    model.config.image_aspect_ratio = image_aspect_ratio
    model.config.use_cache = False
    model.config.image_grid_pinpoints = None
    model.config.freeze_mm_mlp_adapter = True

    model = model.eval()
    return tokenizer, model, image_processor