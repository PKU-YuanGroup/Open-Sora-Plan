import os
import torch
import argparse
import pandas as pd
from opensora.utils.utils import set_seed
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPImageProcessor
from torch import nn, einsum
from PIL import Image
from opensora.eval.mps import trainer
from opensora.eval.general import get_meta_for_step2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
 
    args = parser.parse_args()
    return args

metric = [
    'Category', 
    'Challenge', 
    ]

def show_metric(meta_dict_by_id, metric):
    df = pd.DataFrame(meta_dict_by_id.values())
    metrics = list(set(metric) & set(df.columns))
    for metric in metrics:
        tags = df[metric].unique()
        print(f'Metric: {metric}')
        for tag in tags:
            data = df[df[metric]==tag]['reward']
            print(f"\t{tag:<30}, {data.mean()}+-{data.std()}")
        print('-'*50)
    print(f"{'Overall':<30}, {df['reward'].mean()}+-{df['reward'].std()}")


@torch.no_grad()
def infer_one_sample(image, prompt, clip_model, clip_processor, tokenizer, device, condition=None):
    def _process_image(image):
        image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values
    
    def _tokenize(caption):
        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids

    image_input = _process_image(image).to(device)
    text_input = _tokenize(prompt).to(device)
    condition = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."
    condition_batch = _tokenize(condition).repeat(text_input.shape[0], 1).to(device)

    with torch.no_grad():
        text_f, text_features = clip_model.model.get_text_features(text_input)

        image_f = clip_model.model.get_image_features(image_input.half())
        condition_f, _ = clip_model.model.get_text_features(condition_batch)

        sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.3, 0, float('-inf'))
        mask = mask.repeat(1,image_f.shape[1],1)
        image_features = clip_model.cross_model(image_f, text_f, mask.half())[:,0,:]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_score = clip_model.logit_scale.exp() * text_features @ image_features.T
    return image_score


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, rank=0, device_specific=False)
    args.batch_size = 1
    assert args.batch_size == 1

    device = torch.device("cuda:0")
    image_processor = CLIPImageProcessor.from_pretrained(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    args.model_path = os.path.join(args.model_path, "MPS_overall_checkpoint.pth")
    # import ipdb;ipdb.set_trace()
    model = torch.load(args.model_path, map_location='cpu')
    model.model.text_model.eos_token_id = 2
    model = model.eval().to(device)
    model = model.to(torch.float32)

    batch_size = args.batch_size
    meta_info, meta_dict_by_id = get_meta_for_step2(args.prompt_type)
    meta_info = [meta_info[i:i + batch_size] for i in range(0, len(meta_info), batch_size)]

    
    for batch in tqdm(meta_info):
        prompt = [i['Prompts'] for i in batch]
        ids = [i['id'] for i in batch]
        img_list = [os.path.join(args.image_dir, f'{i}.jpg') for i in ids]
        with torch.no_grad():
            rewards = infer_one_sample(img_list[0], prompt[0], model, image_processor, tokenizer, device)
            # print(f"rewards = {rewards}")
        for index, id_ in enumerate(ids):
            meta_dict_by_id[id_].update({'reward': rewards[index]})

    show_metric(meta_dict_by_id, metric)