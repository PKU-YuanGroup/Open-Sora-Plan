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
import clip
from PIL import Image
from opensora.eval.general import get_meta_for_step2
from opensora.eval.laionaesv2.predictor import MLP, normalized

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
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


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, rank=0, device_specific=False)
    args.batch_size = 1
    assert args.batch_size == 1

    device = torch.device("cuda:0")

    model2, preprocess = clip.load("ViT-L/14", device=device, download_root=args.model_path)  #RN50x64   
    model2 = model2.eval().to(device)

    args.model_path = os.path.join(args.model_path, "sac+logos+ava1-l14-linearMSE.pth")

    model = MLP(768)
    s = torch.load(args.model_path)  
    model.load_state_dict(s)
    model.to(device)
    model.eval()

    batch_size = args.batch_size
    meta_info, meta_dict_by_id = get_meta_for_step2(args.prompt_type)
    meta_info = [meta_info[i:i + batch_size] for i in range(0, len(meta_info), batch_size)]

    
    for batch in tqdm(meta_info):
        ids = [i['id'] for i in batch]
        img_list = [Image.open(os.path.join(args.image_dir, f'{i}.jpg')) for i in ids]
        images = torch.stack([preprocess(i) for i in img_list]).to(device)

        with torch.no_grad():
            image_features = model2.encode_image(images)
        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        rewards = model(torch.from_numpy(im_emb_arr).to(device).float())

        for index, id_ in enumerate(ids):
            meta_dict_by_id[id_].update({'reward': rewards[index]})

    show_metric(meta_dict_by_id, metric)