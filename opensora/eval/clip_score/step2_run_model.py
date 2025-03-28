import os
import torch
import argparse
import pandas as pd
from opensora.utils.utils import set_seed
from copy import deepcopy
import numpy as np
from PIL import Image
from tqdm import tqdm
from opensora.eval.general import get_meta_for_step2
from torchmetrics.multimodal import CLIPScore

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, rank=0, device_specific=False)

    device = torch.device('cuda:0')

    model = CLIPScore(model_name_or_path=args.model_path).to('cuda')
    model = model.eval()
    model = model.to(device)
    model = model.to(torch.float32)

    def calculate_clip_score(images, prompts):
        images_int = np.stack([np.array(Image.open(i).convert("RGB")) for i in images]).astype("uint8")
        # images_int = (images * 255).astype("uint8")
        clip_score = model(torch.from_numpy(images_int).permute(0, 3, 1, 2).to('cuda'), prompts).detach()
        return round(float(clip_score), 4)

    batch_size = args.batch_size
    meta_info, meta_dict_by_id = get_meta_for_step2(args.prompt_type)
    meta_info = [meta_info[i:i + batch_size] for i in range(0, len(meta_info), batch_size)]

    scores = []
    for batch in tqdm(meta_info):
        prompt = [i['Prompts'] for i in batch]
        ids = [i['id'] for i in batch]
        img_list = [os.path.join(args.image_dir, f'{i}.jpg') for i in ids]

        score = calculate_clip_score(img_list, prompt)
        scores.append(score)

    scores = np.array(scores)
    print(f'Overall: {scores.mean():4f}+-{scores.std():4f}')

