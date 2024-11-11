from glob import glob
from opensora.models.diffusion.opensora_v1_5.modeling_opensora import OpenSoraT2V_v1_5
from tqdm import tqdm
from copy import deepcopy
import torch
import os

save_folder = 'merge_model'
os.makedirs(save_folder, exist_ok=True)
# model_paths = [
#     'mmdit13b_dense_rf_bs8192_lr1e-4_max1x256x256_min1x192x192_emaclip99_recap_coyo_merge_1025/checkpoint-99971/model', 
#     'mmdit13b_dense_rf_bs8192_lr1e-4_max1x256x256_min1x192x192_emaclip99_recap_coyo_merge_1025/checkpoint-99971/model_ema', 
#     'mmdit13b_dense_rf_bs8192_lr1e-4_max1x256x256_min1x192x192_emaclip99_recap_coyo_merge_1025/checkpoint-101103/model', 
#     'mmdit13b_dense_rf_bs8192_lr1e-4_max1x256x256_min1x192x192_emaclip99_recap_coyo_merge_1025/checkpoint-101103/model_ema', 
# ]
model_paths = list(glob('mmdit13b_dense_rf_bs8192_lr1e-4_max1x256x256_min1x192x192_emaclip99_recap_coyo_merge_1025/checkpoint-*/model*'))
model_paths = sorted(model_paths, key=lambda x: int(x.split("-")[-1].split('/')[0]))
print(model_paths)
num_model = len(model_paths)
first_model_path = model_paths[0]
model = OpenSoraT2V_v1_5.from_pretrained(first_model_path, torch_dtype=torch.float32).eval()
state_dict = model.state_dict()
avg_state_dict = {k: v/num_model for k, v in state_dict.items()}

ema_decays = [0.9, 0.95, 0.99, 0.995]
state_dicts = {i: deepcopy(state_dict) for i in ema_decays}
model_paths = model_paths[1:]
for i, model_path in enumerate(tqdm(model_paths)):
    model = OpenSoraT2V_v1_5.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    model_state_dict = model.state_dict()

    print('avg_state_dict')
    avg_state_dict = {k: v + model_state_dict[k]/num_model for k, v in tqdm(avg_state_dict.items())}

    num_ema_decay = len(ema_decays)
    print('ema_decays')
    for j in tqdm(range(num_ema_decay)):
        tmp_ema_decay = ema_decays[j]
        tmp_state_dict = state_dicts[tmp_ema_decay]
        for k, v in tqdm(model_state_dict.items()):
            tmp_state_dict[k] = tmp_ema_decay * tmp_state_dict[k] + (1-tmp_ema_decay) * v
        state_dicts[tmp_ema_decay] = tmp_state_dict


for j in range(num_ema_decay):
    tmp_ema_decay = ema_decays[j]
    tmp_state_dict = state_dicts[tmp_ema_decay]
    model.load_state_dict(tmp_state_dict)
    model.save_pretrained(f'{save_folder}/ema_{tmp_ema_decay}')


model.load_state_dict(avg_state_dict)
model.save_pretrained(f'{save_folder}/avg_model')
