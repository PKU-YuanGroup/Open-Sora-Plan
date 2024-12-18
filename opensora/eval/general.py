import pandas as pd
from copy import deepcopy
import json

meta_path_dict = {
    'GenAI': "opensora/eval/eval_prompts/GenAI-Image-527/genai_image.json", 
    "DrawBench": "opensora/eval/eval_prompts/DrawBench.tsv", 
    "PartiPrompts": "opensora/eval/eval_prompts/PartiPrompts.tsv",
}

def get_meta(prompt_type):
    meta_path = meta_path_dict[prompt_type]
    if meta_path.endswith('tsv'):
        meta_info = pd.read_csv(meta_path, sep='\t')
        meta_info = meta_info.rename(columns={'Prompt': 'Prompts'})
        meta_info['id'] = meta_info.index.map(lambda x: f"{x:08d}")
        return meta_info.to_dict(orient='records')
    else:
        with open(meta_path, 'r') as f:
            meta_info = json.load(f)

        ret_meta_info = []
        for v in meta_info.values():
            del v['models']
            del v['prompt in Chinese']
            v['Prompts'] = deepcopy(v['prompt'])
            del v['prompt']
            v['Category'] = 'No Category'
            v['id'] = f"{int(v['id']):05d}"
            ret_meta_info.append(v)
        return ret_meta_info



def get_meta_for_step2(prompt_type):
    meta_path = meta_path_dict[prompt_type]
    if meta_path.endswith('tsv'):
        meta_info = pd.read_csv(meta_path, sep='\t')
        meta_info = meta_info.rename(columns={'Prompt': 'Prompts'})
        meta_info['id'] = meta_info.index.map(lambda x: f"{x:08d}")
        meta_info = meta_info.to_dict(orient='records')
        meta_info_cp = deepcopy(meta_info)
        meta_dict_by_id = {}
        for i in meta_info:
            key = i['id']
            del i['id']
            meta_dict_by_id[key] = i
        return meta_info_cp, meta_dict_by_id
    else:
        with open(meta_path, 'r') as f:
            meta_info = json.load(f)

        ret_meta_info = []
        for v in meta_info.values():
            del v['models']
            del v['prompt in Chinese']
            v['Prompts'] = deepcopy(v['prompt'])
            del v['prompt']
            v['Category'] = 'No Category'
            ret_meta_info.append(v)

        ret_meta_info_cp = deepcopy(ret_meta_info)
        meta_dict_by_id = {}
        for i in ret_meta_info:
            key = i['id']
            del i['id']
            meta_dict_by_id[key] = i
        
        return ret_meta_info_cp, meta_dict_by_id