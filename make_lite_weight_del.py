from opensora.models.diffusion.opensora_t2i.modeling_opensora import OpenSoraT2I
from tqdm import tqdm
from copy import deepcopy
import os


num_layers = [
    [1] * 12 + [0] * 0 + [1] * 0, 
    [0] * 1 + [1] * 0, 
    [1] * 12 + [0] * 0 + [1] * 0, 
]

name_map = {}
for tr_stage, tr_layers in enumerate(num_layers):
    target_stage_name = f'transformer_blocks.{tr_stage}'
    for i in range(len(tr_layers)):
        is_del = tr_layers[i] == 0
        if is_del:
            name_map[f'{target_stage_name}.{i}'] = 'del'
        else:
            name_map[f'{target_stage_name}.{i}'] = f'{target_stage_name}.{sum(tr_layers[:i+1])-1}'

print(name_map)

model = OpenSoraT2I.from_pretrained("t2i_ablation_arch/mixnorm/checkpoint-276000/model_ema")
state_dict = model.state_dict()
new_state_dict = {}
model_config = deepcopy(model.config)
# import ipdb;ipdb.set_trace()
for k, v in tqdm(state_dict.items()):
    if 'transformer_blocks' in k:
        name_split = k.split('.')
        block_name = '.'.join(name_split[:3])
        if name_map[block_name] == 'del':
            print(f'del block_name: {block_name}')
            continue
        elif name_map[block_name] != 'del':
            new_block_name = name_map[block_name]
            k = new_block_name + '.' + '.'.join(name_split[3:])
        else:
            import ipdb;ipdb.set_trace()
            raise ValueError
    new_state_dict[k] = v
model_config_del = deepcopy(model_config)
model_config_del = {k: v for k, v in model_config_del.items()}
model_config_del['num_layers'] = [sum(i) for i in num_layers]
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
model.save_pretrained(f'mixnorm_delmid')