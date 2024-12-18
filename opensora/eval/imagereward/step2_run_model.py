import os
import torch
from opensora.eval.imagereward import RM
import argparse
import pandas as pd
from opensora.utils.utils import set_seed
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from opensora.eval.general import get_meta_for_step2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
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

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, rank=0, device_specific=False)

    model = RM.load(model_path=args.model_path, tokenizer_path=args.tokenizer_path, name="ImageReward-v1.0")
    model = model.to(torch.float32)

    batch_size = args.batch_size
    meta_info, meta_dict_by_id = get_meta_for_step2(args.prompt_type)
    meta_info = [meta_info[i:i + batch_size] for i in range(0, len(meta_info), batch_size)]

    
    for batch in tqdm(meta_info):
        prompt = [i['Prompts'] for i in batch]
        ids = [i['id'] for i in batch]
        img_list = [os.path.join(args.image_dir, f'{i}.jpg') for i in ids]
        with torch.no_grad():
            _, rewards = model.inference_rank(prompt, img_list)
            # print(f"rewards = {rewards}")
        for index, id_ in enumerate(ids):
            meta_dict_by_id[id_].update({'reward': rewards[index]})

    show_metric(meta_dict_by_id, metric)