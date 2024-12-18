import os
import torch
import argparse
import pandas as pd
from opensora.utils.utils import set_seed
from copy import deepcopy
import numpy as np
from PIL import Image
from tqdm import tqdm
from opensora.eval.vqascore import t2v_metrics
from opensora.eval.general import get_meta_for_step2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, rank=0, device_specific=False)

    device = torch.device('cuda:0')
    model =  t2v_metrics.VQAScore(model=args.model_path)
    model = model.eval()
    model = model.to(device)
    model = model.to(torch.float32)

    batch_size = args.batch_size
    meta_info, meta_dict_by_id = get_meta_for_step2(args.prompt_type)
    meta_info = [meta_info[i:i + batch_size] for i in range(0, len(meta_info), batch_size)]

    scores = []
    for batch in tqdm(meta_info):
        prompt = [i['Prompts'] for i in batch]
        ids = [i['id'] for i in batch]
        img_list = [os.path.join(args.image_dir, f'{i}.jpg') for i in ids]

        dataset = [dict(images=[i], texts=[j]) for i, j in zip(img_list, prompt)]
        assert args.batch_size == len(dataset)
        score = model.batch_forward(dataset=dataset, batch_size=args.batch_size)
        scores.extend(score.detach().flatten().float().cpu().tolist())

    scores = np.array(scores)
    print(f'Overall: {scores.mean()*100:3f}+-{scores.std()*100:3f}')

