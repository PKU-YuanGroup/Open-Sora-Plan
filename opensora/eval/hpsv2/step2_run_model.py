import os
import torch
import argparse
import pandas as pd
from opensora.utils.utils import set_seed
from copy import deepcopy
import numpy as np
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer

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

if __name__ == "__main__":
    args = get_args()
    args.batch_size = 1
    assert args.batch_size == 1
    set_seed(args.seed, rank=0, device_specific=False)

    device = torch.device('cuda:0')
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        os.path.join(args.tokenizer_path, 'open_clip_pytorch_model.bin'),
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )

    cp = os.path.join(args.model_path, "HPS_v2.1_compressed.pt")
    
    checkpoint = torch.load(cp, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    batch_size = args.batch_size
    meta_info, meta_dict_by_id = get_meta_for_step2(args.prompt_type)
    meta_info = [meta_info[i:i + batch_size] for i in range(0, len(meta_info), batch_size)]

    
    for batch in tqdm(meta_info):
        prompt = [i['Prompts'] for i in batch]
        ids = [i['id'] for i in batch]
        img_list = [os.path.join(args.image_dir, f'{i}.jpg') for i in ids]

        image = preprocess_val(Image.open(img_list[0])).unsqueeze(0).to(device=device, non_blocking=True)
        text = tokenizer(prompt).to(device=device, non_blocking=True)
        # Calculate the HPS
        with torch.no_grad():
            outputs = model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T

            hps_score = torch.diagonal(logits_per_image).cpu().numpy()

        for index, id_ in enumerate(ids):
            meta_dict_by_id[id_].update({'reward': hps_score[index]})

    show_metric(meta_dict_by_id, metric)