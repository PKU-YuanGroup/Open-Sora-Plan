from opensora.models.diffusion.opensora_t2i.modeling_opensora import OpenSoraT2I
from tqdm import tqdm
from copy import deepcopy
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    return args

args = parse_args()
model_path = args.model_path
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
model = OpenSoraT2I.from_pretrained(model_path)
state_dict = model.state_dict()

model_config = deepcopy(model.config)
num_layers = [12, 1, 12]
for tr_stage, tr_layers in enumerate(num_layers):
    for tr_layer in tqdm(range(tr_layers)):
        if os.path.exists(f'{save_path}/del_s{tr_stage}_l{tr_layer}'):
            continue
        new_state_dict = {}
        print(f'del layer {tr_layer} of stage {tr_stage}')
        for k, v in tqdm(state_dict.items()):
            target_stage_name = f'transformer_blocks.{tr_stage}'
            if target_stage_name in k:
                name_split = k.split('.')
                layer_i = int(name_split[2])
                if layer_i == tr_layer:
                    print(f'del k: {k}')
                    continue
                if layer_i > tr_layer:
                    layer_i = layer_i - 1
                    name_split[2] = str(layer_i)
                    print(f'origin k: {k} -> ', end='')
                    k = '.'.join(name_split)                            
                    print(k)
                else:
                    print(f'do not change k: {k}')
                    pass
            new_state_dict[k] = v
        model_config_del = deepcopy(model_config)
        model_config_del = {k: v for k, v in model_config_del.items()}
        model_config_del['num_layers'][tr_stage] = model_config_del['num_layers'][tr_stage] - 1
        del model_config_del['decay']
        del model_config_del['inv_gamma']
        del model_config_del['min_decay']
        del model_config_del['optimization_step']
        del model_config_del['power']
        del model_config_del['update_after_step']
        del model_config_del['use_ema_warmup']
        # import ipdb;ipdb.set_trace()
        print(model_config_del)
        model = OpenSoraT2I.from_config(model_config_del)
        model.load_state_dict(new_state_dict)
        model.save_pretrained(f'{save_path}/del_s{tr_stage}_l{tr_layer}')