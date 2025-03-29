# Evaluate on GenAI-Bench-Image (with 527 prompt) using a specific model
# Example scripts to run:
# VQAScore: python genai_image_eval.py --model clip-flant5-xxl
# CLIPScore: python genai_image_eval.py --model openai:ViT-L-14-336
# GPT4o VQAScore: python genai_image_eval.py --model gpt-4o
import argparse
import os
from opensora.eval.vqascore import t2v_metrics
import json
import torch
import numpy as np
from opensora.eval.vqascore.t2v_metrics.dataset import GenAIBench_Image



tag_groups = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
    'overall': ['basic', 'advanced', 'all']
}

def show_performance_per_skill(our_scores, dataset, items_name='images', prompt_to_items_name='prompt_to_images', print_std=False):
    tag_result = {}
    tag_file = f"{dataset.meta_dir}/genai_skills.json"
    tags = json.load(open(tag_file))
    items = getattr(dataset, items_name)
    prompt_to_items = getattr(dataset, prompt_to_items_name)
    items_by_model_tag = {}
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                model = items[image_idx]['model']
                if model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][model] = []
                items_by_model_tag[tag][model].append(image_idx)
    
    for tag in tags:
        # print(f"Tag: {tag}")
        tag_result[tag] = {}
        for model in items_by_model_tag[tag]:
            our_scores_mean = our_scores[items_by_model_tag[tag][model]].mean()
            our_scores_std = our_scores[items_by_model_tag[tag][model]].std()
            # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
            tag_result[tag][model] = {
                'metric': {'mean': our_scores_mean, 'std': our_scores_std},
            }
        # print()
        
    # print("All")
    tag_result['all'] = {}
    all_models = items_by_model_tag[tag]
    for model in all_models:
        all_model_indices = set()
        for tag in items_by_model_tag:
            all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][model]))
        all_model_indices = list(all_model_indices)
        our_scores_mean = our_scores[all_model_indices].mean()
        our_scores_std = our_scores[all_model_indices].std()
        # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
        tag_result['all'][model] = {
            'metric': {'mean': our_scores_mean, 'std': our_scores_std},
        }
    
    for tag_group in tag_groups:
        for score_name in ['metric']:
            print(f"Tag Group: {tag_group} ({score_name} performance)")
            tag_header = f"{'Model':<17}" + " ".join([f"{tag:<17}" for tag in tag_groups[tag_group]])
            print(tag_header)
            for model_name in all_models:
                if print_std:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f}+-{tag_result[tag][model_name][score_name]['std']:.2f}" for tag in tag_groups[tag_group]]
                else:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f}" for tag in tag_groups[tag_group]]
                detailed_scores = " ".join([f"{score:<17}" for score in detailed_scores])
                model_scores = f"{model_name:<17}" + detailed_scores
                print(model_scores)
            print()
        print()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
 
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    image_dir = args.image_dir
    meta_dir = args.meta_dir

    dataset = GenAIBench_Image(root_dir=image_dir, meta_dir=meta_dir)

    model = args.model_path
    device = torch.device('cuda:0')
    score_func = t2v_metrics.get_score_model(model=model, device=device)

    kwargs = {}
    scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        
    
    ### Get performance per skill
    our_scores = scores.mean(axis=1)
    show_performance_per_skill(our_scores, dataset, print_std=True)    
    
if __name__ == "__main__":
    main()
