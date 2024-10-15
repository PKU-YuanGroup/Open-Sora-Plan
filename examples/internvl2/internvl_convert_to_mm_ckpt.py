import os
from copy import deepcopy
import stat

import torch
from transformers import AutoModelForCausalLM


def load_from_hf(_load_dir):
    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(_load_dir, device_map='cpu', trust_remote_code=True)
    print(hf_model)
    return hf_model.state_dict()


def convert_hg_to_mm(_state_dict, _num_layers):
    new_dict = {}
    for key, value in _state_dict.items():
        new_key = None
        # 权重映射
        if key.startswith('vision_model'):
            new_key = key.replace('vison_model', 'image_encoder.encoder')
            new_key = new_key.replace('embeddings', 'embeddings')
            new_key = new_key.replace('norm1', 'input_layernorm')
            new_key = new_key.replace('norm2', 'pre_mlp_layernorm')
            new_key = new_key.replace('attn', 'self_attention')
            new_key = new_key.replace('inner_attn', 'core_attention')
            new_key = new_key.replace('q_norm', 'q_layernorm')
            new_key = new_key.replace('k_norm', 'k_layernorm')
            new_key = new_key.replace('qkv', 'linear_qkv')
            new_key = new_key.replace('proj', 'linear_proj')
            new_key = new_key.replace('mlp.fc1', 'mlp.linear_fc1')
            new_key = new_key.replace('mlp.fc2', 'mlp.linear_fc2')

        elif key.startswith('language_model'):
            new_key = key.replace('language_model.model.tok_embeddings', 'text_decoder.embedding.word_embeddings')
            new_key = new_key.replace('language_model.model.norm', 'text_decoder.decoder.final_layernorm')
            new_key = new_key.replace('language_model.output', 'text_decoder.output_layer')
            new_key = new_key.replace('language_model.model', 'text_decoder.decoder')
            new_key = new_key.replace('attention_norm', 'input_layernorm')
            new_key = new_key.replace('attention.wo', 'self_attention.linear_proj')
            new_key = new_key.replace('attention.wqkv', 'self_attention.linear_qkv')
            new_key = new_key.replace('ffn_norm', 'pre_mlp_layernorm')
            new_key = new_key.replace('feed_forward.w1', 'mlp.linear_fc1_gate')
            new_key = new_key.replace('feed_forward.w3', 'mlp.linear_fc1_up')
            new_key = new_key.replace('feed_forward.w2', 'mlp.linear_fc2')

        elif key.startswith('mlp1'):
            new_key = key.replace('mlp1', 'vit_proj')

        # 打印映射过程
        print(f'mapping {key} to {new_key}')

        new_dict[new_key] = value

    # 合并w1和w3权重
    for i in range(_num_layers):
        gate_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_gate.weight'
        up_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1_up.weight'
        fc1_name = f'text_decoder.decoder.layers.{i}.mlp.linear_fc1.weight'

        # 合并 w1 和 w3
        if gate_name in new_dict.keys():
            gate_proj_weight = new_dict[gate_name]
        if up_name in new_dict.keys():
            up_proj_weight = new_dict[up_name]
        linear_fc1 = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
        new_dict[fc1_name] = linear_fc1

        # 移除合并前的权重
        new_dict.pop(gate_name)
        new_dict.pop(up_name)

        print(f'merge {gate_name} and {up_name} to {fc1_name}')

    return new_dict


def split_by_pp(_state_dict, _num_layers, _pipeline_layer_index=None):
    if _pipeline_layer_index is None:
        return [_state_dict, ], _
    return_dicts = []
    copy_dict = deepcopy(_state_dict)
    for pp_rank, _ in enumerate(_pipeline_layer_index):
        is_first = False
        is_last = False
        if pp_rank == 0:
            is_first = True
        elif pp_rank == len(_pipeline_layer_index) - 1:
            is_last = True

        pp_start_index = _pipeline_layer_index[pp_rank]
        if is_last:
            pp_end_index = _num_layers
        else:
            pp_end_index = _pipeline_layer_index[pp_rank + 1]

        new_dict = {}
        for key, value in _state_dict.items():
            if key.startswith('image_encoder'):
                if is_first:
                    new_dict[key] = value
                    copy_dict.pop(key)
            if key.startswith('vit_proj'):
                if is_first:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.embedding'):
                if is_first:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.final_layernorm'):
                if is_last:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.output_layer'):
                if is_last:
                    new_dict[key] = value
                    copy_dict.pop(key)
            elif key.startswith('text_decoder.decoder.layers.'):
                layer = int(key.split('.')[3])
                if layer >= pp_start_index and layer < pp_end_index:
                    new_layer = layer - pp_start_index
                    new_key = key.replace(str(layer), str(new_layer))
                    new_dict[new_key] = value
                    copy_dict.pop(key)
        return_dicts.append(new_dict)
    return return_dicts, copy_dict


def save_by_pp(_state_dicts, _save_dir, _latest_checkpointed_iteration='release', _exists_ok=False):
    if os.path.exists(_save_dir):
        if not _exists_ok:
            print(f'save dir: {_save_dir} exists, please check.')
            return
    else:
        os.makedirs(_save_dir)
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(_save_dir, 'latest_checkpointed_iteration.txt'), flags, mode), 'w') as fout:
        fout.write(_latest_checkpointed_iteration)

    if _latest_checkpointed_iteration == 'release':
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(_latest_checkpointed_iteration)

    for pp_rank, state_dic in enumerate(_state_dicts):
        tp_rank = 0
        os.makedirs(os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'))
        save_path = os.path.join(_save_dir, directory, f'mp_rank_{tp_rank:02d}_{pp_rank:03d}', 'model_optim_rng.pt')
        save_dict = {}
        save_dict['model'] = state_dic
        torch.save(save_dict, save_path)


if __name__ == "__main__":
    hg_ckpt_dir = "InternVL2-8B" # huggingface权重目录
    mm_save_dir = 'InternVL2-8B_pp4'  # 转换后权重保持目录
    pipeline_layer_index = [0, 3, 13, 23]
    num_layers = 32

    state_dict = load_from_hf(_load_dir=hg_ckpt_dir)
    state_dict = convert_hg_to_mm(state_dict, num_layers)
    state_dicts, _ = split_by_pp(state_dict, num_layers, pipeline_layer_index)
    save_by_pp(state_dicts, mm_save_dir, _exists_ok=True)