from collections.abc import Iterable
import os

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
    if "ema" in ckpt_dict:  # supports checkpoints from train.py
        ckpt_dict = ckpt_dict["ema"]

    missing_keys, unexpected_keys = model.load_state_dict(ckpt_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
