from collections.abc import Iterable
import os
from megatron.core import mpu
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

import safetensors


def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    if not isinstance(model, nn.Module):
        raise AssertionError("model must be nn.Module")

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", True):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


def load_checkpoint(model, ckpt_path):
    # TODO: 使用统一的接口进行模型保存/加载管理
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")

    if ckpt_path.endswith("pt") or ckpt_path.endswith("pth"):
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    elif ckpt_path.endswith(".safetensors"):
        ckpt_dict = safetensors.torch.load_file(ckpt_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

    if "pos_embed_temporal" in ckpt_dict:
        del ckpt_dict["pos_embed_temporal"]
    if "pos_embed" in ckpt_dict:
        del ckpt_dict["pos_embed"]
    if "ema_state_dict" in ckpt_dict:  # supports checkpoints from train.py
        ckpt_dict = ckpt_dict["ema_state_dict"]

    replacements = {
        # vae
        "module.": "",
        # dit
        "transformer_blocks": "videodit_blocks",
        "attn1": "self_atten",
        "attn2": "cross_atten",
        "to_q": "proj_q",
        "to_k": "proj_k",
        "to_v": "proj_v",
        "to_out.0": "proj_out",
        "to_out.1": "dropout"
    }

    new_ckpt_dict = {}
    for model_key in ckpt_dict:
        new_key = model_key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        new_ckpt_dict[new_key] = ckpt_dict[model_key]
    ckpt_dict = new_ckpt_dict

    suffixes_1 = ["atten.proj_q.weight", "atten.proj_q.bias", "atten.proj_k.weight", "atten.proj_k.bias",
                  "atten.proj_v.weight", "atten.proj_v.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    suffixes_2 = ["atten.proj_out.weight", "ff.net.2.weight"]
    for key, value in ckpt_dict.items():
        if isinstance(value, torch.Tensor):
            if any(key.endswith(suffix) for suffix in suffixes_1):
                ckpt_dict[key] = torch.chunk(value, mpu.get_tensor_model_parallel_world_size(), dim=0)[
                    mpu.get_tensor_model_parallel_rank()]
                # print(f"Key1: {key}, Shape: {ckpt_dict[key].shape}")

            if any(key.endswith(suffix) for suffix in suffixes_2):
                ckpt_dict[key] = torch.chunk(value, mpu.get_tensor_model_parallel_world_size(), dim=1)[
                    mpu.get_tensor_model_parallel_rank()]
                # print(f"Key2: {key}, Shape: {ckpt_dict[key].shape}")
        else:
            # print(f"Key: {key}, Type: {type(value)}")
            pass

    missing_keys, unexpected_keys = model.load_state_dict(ckpt_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
