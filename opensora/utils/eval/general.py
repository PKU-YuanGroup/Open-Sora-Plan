import pandas as pd
from copy import deepcopy
import json

meta_path_dict = {
    'GenAI527': "opensora/eval/eval_prompts/GenAI527/genai_image.json", 
    'GenAI1600': "opensora/eval/eval_prompts/GenAI1600/genai_image.json", 
    "DrawBench": "opensora/eval/eval_prompts/DrawBench.tsv", 
    "PartiPrompts": "opensora/eval/eval_prompts/PartiPrompts.tsv",
    "DALLE3": "opensora/eval/eval_prompts/DALLE3.txt",
    "DOCCI-Test-Pivots": "opensora/eval/eval_prompts/DOCCI-Test-Pivots.txt",
    "Gecko-Rel": "opensora/eval/eval_prompts/Gecko-Rel.txt", 
    "COCO2017": "opensora/eval/eval_prompts/COCO2017.json", 
    "ImageNet": "opensora/eval/eval_prompts/ImageNet.json", 
}

def get_meta(prompt_type):
    '''
    [
        {
            "Prompt": "a photo of a cat",
            "Category": "", 
            "id": "", 
        },
        ...
    ]
    '''
    meta_path = meta_path_dict[prompt_type]
    if meta_path.endswith('tsv'):
        meta_info = pd.read_csv(meta_path, sep='\t')
        meta_info = meta_info.rename(columns={'Prompt': 'Prompts'})
        meta_info['id'] = meta_info.index.map(lambda x: f"{x:08d}")
        return meta_info.to_dict(orient='records')
    elif meta_path.endswith('json'):
        with open(meta_path, 'r') as f:
            meta_info = json.load(f)

        ret_meta_info = []
        for v in meta_info.values():
            if 'models' in v: del v['models']
            if 'prompt in Chinese' in v: del v['prompt in Chinese']
            v['Prompts'] = deepcopy(v['prompt'])
            if 'prompt' in v: del v['prompt']
            v['Category'] = 'No Category'
            v['id'] = f"{int(v['id']):09d}"
            ret_meta_info.append(v)
        return ret_meta_info
    else:
        with open(meta_path, 'r') as f:
            meta_info = f.readlines()
        meta_info = [dict(Prompts=text.strip(), id=f"{idx:09d}", Category='No Category') for idx, text in enumerate(meta_info)]
        return meta_info



def get_meta_for_step2(prompt_type):
    '''
    {
        "id" :{
            "Prompt": "a photo of a cat",
            "Category": "", 
        },
        ...
    }
    '''
    meta_info = get_meta(prompt_type)
    meta_info_cp = deepcopy(meta_info)
    meta_dict_by_id = {}
    for i in meta_info:
        key = i['id']
        del i['id']
        meta_dict_by_id[key] = i
    return meta_info_cp, meta_dict_by_id

if __name__ == '__main__':
    prompt_type = 'ImageNet'
    meta_info = get_meta(prompt_type)
    print(meta_info)