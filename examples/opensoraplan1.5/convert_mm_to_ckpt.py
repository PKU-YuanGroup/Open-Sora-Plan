import os
import argparse
import re
from pathlib import Path
from typing import Optional, Sequence, Any, Tuple, Dict
from copy import deepcopy

import torch
import mindspeed.megatron_adaptor


def get_ckpt_path(path: str) -> Tuple[Path, Optional[str]]:
    """ 判断path内是否有iteration信息，如果有，去找对应目录，判断对应目录下是否有符合要求的目录；如果没有，则判断本目录是否已经满足要求 """
    path = Path(path)
    files = [item.name for item in path.iterdir() if item.is_file()]
    if 'latest_checkpointed_iteration.txt' in files:
        iteration = path.joinpath('latest_checkpointed_iteration.txt').read_text('utf-8')
        if iteration.isdigit():
            path_ = path.joinpath(f'iter_{int(iteration):07d}')
        else:
            path_ = path.joinpath(iteration)
        if not Path.exists(path_):
            raise FileNotFoundError(f'Path `{path_}` is not exists.')
        return path_, iteration
    else:
        pattern = re.compile(r'mp_?([a-zA-Z0-9]+)?_rank_(\d{2})(?:_(\d{3}))?')
        dirs = [item.name for item in path.iterdir() if item.is_dir()]
        for dir_name in dirs:
            if not pattern.match(dir_name):
                raise ValueError(f'Found unexpected dir `{dir_name}`')
        return path, None


def get_loaded_ckpt_tp_pp(path: Path, extra_model_name: str = None):
    """ 自动获取待加载的ckpt的tp和pp """
    dirs = [item.name for item in path.iterdir() if item.is_dir()]
    pattern = re.compile(r'mp_?([a-zA-Z0-9]+)?_rank_(\d{2})(?:_(\d{3}))?')
    tp_size, pp_size = 0, 0
    for dir_name in dirs:
        match = pattern.match(dir_name)
        if match:
            if match.group(1) != extra_model_name:
                continue
            tp_size = max(int(match.group(2)) + 1, tp_size)
            pp_size = max(int(match.group(3)) + 1 if match.group(3) is not None else 1, pp_size)
        else:
            raise ValueError(f'Unexpected dir: {dir_name}')
    if tp_size <= 0 or pp_size <= 0:
        raise ValueError(f'tp_size ({tp_size}) or pp_size ({pp_size}) must greater than 0.')
    return tp_size, pp_size


def load_ckpt(
    path, tp_size, pp_size, extra_model_name=None, ema=False
) -> Tuple[Dict[Tuple[int, int], Dict[str, torch.Tensor]], Dict[str, Any]]:
    """ 加载ckpt，返回{(int, int): {'layers': Tensor}}, {原来state_dict中除了'model'以外的所有信息} """
    ckpts, params = {}, None
    model_key = 'model' if not ema else 'ema_model'
    for tp in range(tp_size):
        for pp in range(pp_size):
            model_name = '' if extra_model_name is None else '_' + extra_model_name
            pp_suffix = '' if pp_size == 1 else f'_{pp:03d}'
            dir_name = 'mp' + model_name + f'_rank_{tp:02d}' + pp_suffix
            state_dict = torch.load(path / dir_name / 'model_optim_rng.pt', map_location='cpu')
            print(f"state_dict: {state_dict.keys()}")
            print(f"ema: {ema}")
            print(f"state_dict[model].dtype: {list(state_dict[model_key].values())[0].dtype}")
            ckpts[(tp, pp)] = deepcopy(state_dict[model_key])
            if params is None:
                params = state_dict
                params.pop(model_key)
    return ckpts, params


def calculate_pp_layer_sizes(ckpt_list, startswith):
    """ 用来根据ckpts计算pp_layer_sizes，在merge_by_pp中使用 """
    pp_layer_sizes = [0 for _ in range(len(ckpt_list))]
    j = len(startswith.split('.'))
    for i, ckpt in enumerate(ckpt_list):
        for k in ckpt.keys():
            if k.startswith(startswith):
                pp_layer_sizes[i] = max(pp_layer_sizes[i], int(k.split('.')[j]) + 1)
    return pp_layer_sizes


def _cumulative_sum(pp_layer_sizes):
    """ 计算每个pp_layer的起始layer_index """
    result = [0]  # 初始化结果列表，第一个元素为0
    for num in pp_layer_sizes:
        result.append(result[-1] + num)  # 将当前元素与结果列表的最后一个元素相加，并添加到结果列表中
    return result


def _get_key_startswith_index_and_str(key, prefix_list):
    """ 返回key对应prefix_list中的哪个，并返回这个index和prefix """
    for i, prefix in enumerate(prefix_list):
        if key.startswith(prefix):
            return i, prefix
    return -1, None


def _merge_by_pp(ori_ckpt, new_ckpt, cur_pp_idx, keys_full_prefix_on_pp_layer, pp_layer_start_index):
    for k, v in ori_ckpt.items():
        startswith_list_index, startswith = _get_key_startswith_index_and_str(k, keys_full_prefix_on_pp_layer)
        if startswith_list_index == -1:
            new_ckpt[k] = v
            continue
        ori_layer_num_index = len(keys_full_prefix_on_pp_layer[startswith_list_index].split('.'))
        ori_layer_num = int(k.split('.')[ori_layer_num_index])
        new_layer_num = ori_layer_num + pp_layer_start_index[startswith][cur_pp_idx]
        k_split = k.split('.')
        k_split[ori_layer_num_index] = str(new_layer_num)
        new_key = '.'.join(k_split)
        new_ckpt[new_key] = v


def merge_by_pp(ckpts, tp_size, pp_size, keys_full_prefix_on_pp_layer: Sequence[str]):
    """ 沿着pp合并ckpt """
    new_ckpts = {}
    for tp in range(tp_size):
        ckpt = deepcopy(ckpts[(tp, 0)])
        ckpts_tp = [ckpts[(tp, pp)] for pp in range(pp_size)]
        pp_layer_start_index = {}
        for startswith in keys_full_prefix_on_pp_layer:
            pp_layer_sizes = calculate_pp_layer_sizes(ckpts_tp, startswith)
            pp_layer_start_index[startswith] = _cumulative_sum(pp_layer_sizes)

        for pp in range(1, pp_size):
            ckpt_ = deepcopy(ckpts_tp[pp])
            _merge_by_pp(ckpt_, ckpt, pp, keys_full_prefix_on_pp_layer, pp_layer_start_index)

        new_ckpts[(tp, 0)] = ckpt

    del ckpts
    return new_ckpts


def _merge_by_tp(ori_ckpt, new_ckpt, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1):

    for k, v in ori_ckpt.items():
        is_split = False
        if isinstance(v, torch.Tensor):
            for k_, v_ in keys_part_on_tp_dim_0.items():
                if k.startswith(k_) and any(k.endswith(vv_) for vv_ in v_):
                    new_ckpt[k] = torch.cat((new_ckpt[k], v), dim=0)
                    is_split = True
            for k_, v_ in keys_part_on_tp_dim_1.items():
                if k.startswith(k_) and any(k.endswith(vv_) for vv_ in v_):
                    new_ckpt[k] = torch.cat((new_ckpt[k], v), dim=1)
                    is_split = True
            if not is_split:
                new_ckpt[k] = v
        else:
            print(f"Not support type {type(v)} for key {k}")
            pass


def merge_by_tp(ckpts, tp_size, pp_size, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1):
    """ 沿着tp合并ckpt """
    new_ckpts = {}
    for pp in range(pp_size):
        ckpt = ckpts[(0, pp)]
        ckpts_pp = [ckpts[(tp, pp)] for tp in range(tp_size)]

        for tp in range(1, tp_size):
            ckpt_ = ckpts_pp[tp]
            _merge_by_tp(ckpt_, ckpt, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)

        new_ckpts[(0, pp)] = ckpt

    new_ckpts = deepcopy(new_ckpts)
    del ckpts
    return new_ckpts



def print_keys(ckpts):
    for k, v in ckpts.items():
        print(k)
        for kk, vv in sorted(v.items()):
            info = kk
            if isinstance(vv, torch.Tensor):
                info += f' {vv.shape} {vv.sum()}'
            else:
                info += f' {type(vv)}'
            print(info)
        print('-' * 30)


def convert(load_dir, save_dir, ema=False):

    # pp keys
    keys_full_prefix_on_pp_layer = ['transformer_blocks']
    keys_full_prefix_on_pp_process = ['transformer_blocks']
    keys_full_prefix_on_pp_postprocess = ['scale_shift_table', 'proj_out.weight', 'proj_out.bias']

    # tp keys
    keys_part_on_tp_dim_0 = {
        '': 
        ["attn1.to_q.weight", "attn1.to_q.bias",
        "attn1.to_k.weight", "attn1.to_k.bias",
        "attn1.to_v.weight", "attn1.to_v.bias",
        "attn1.add_q_proj.weight", "attn1.add_q_proj.bias",
        "attn1.add_k_proj.weight", "attn1.add_k_proj.bias",
        "attn1.add_v_proj.weight", "attn1.add_v_proj.bias",
        "net.0.proj.weight", "net.0.proj.bias",
        "linear.weight", "linear.bias"]
    }
    keys_part_on_tp_dim_1 = {
        '': ["attn1.to_out.0.weight", "attn1.to_add_out.weight", "net.2.weight"]
    }

    # mm part
    path, iteration = get_ckpt_path(load_dir)
    if iteration is None:
        iteration = 'release'
    tp_size, pp_size = get_loaded_ckpt_tp_pp(path)
    print(f'Get saved ckpts have {tp_size=} {pp_size=} {iteration=}, prepare to loading.')
    ckpts, params = load_ckpt(path, tp_size, pp_size, ema=ema)
    args_gpt = getattr(params, 'args', None)
    if args_gpt is not None:
        if tp_size != args_gpt.tensor_model_parallel_size:
            raise ValueError(f'tp_size ({tp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.tensor_model_parallel_size}).')
        if pp_size != args_gpt.pipeline_model_parallel_size:
            raise ValueError(f'pp_size ({pp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.pipeline_model_parallel_size}).')
    print('Ckpts loaded.')
    ckpts = merge_by_pp(ckpts, tp_size, pp_size, keys_full_prefix_on_pp_layer)
    print('Ckpts merged by pp.')
    ckpts = merge_by_tp(ckpts, tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Ckpts merged by tp.')
    # print_keys(ckpts)

    final_ckpt = ckpts[(0, 0)]

    print(final_ckpt.keys())

    save_path = os.path.join(save_dir, f"model{'_ema' if ema else ''}.pt")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(final_ckpt, os.path.join(save_dir, f"model{'_ema' if ema else ''}.pt"))


    
if __name__ == "__main__":
    load_dir = '/work/share1/checkpoint/gyy/osp/121x576x1024_node64_tp4_bs1_gc4_lr1e-5_wd1e-2_hq/iter_0003936/'
    save_dir = '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/test_ckpt/test_ac_121_576_hq_03936'
    ema = True
    convert(load_dir, save_dir, ema=ema)
