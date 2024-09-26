from collections.abc import Iterable
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


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
    if 'safetensors' in ckpt_path:
        from safetensors.torch import load_file as safe_load
        ckpt_dict = safe_load(ckpt_path)
    else:
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

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

    missing_keys, unexpected_keys = model.load_state_dict(ckpt_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
