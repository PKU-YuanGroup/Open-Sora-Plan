from collections.abc import Iterable
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from megatron.core import mpu
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
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")

    if ckpt_path.endswith("pt") or ckpt_path.endswith("pth"):
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    elif ckpt_path.endswith(".safetensors"):
        ckpt_dict = safetensors.torch.load_file(ckpt_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    elif "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]

    suffixes_1 = ["attn1.to_q.weight", "attn1.to_q.bias",
                  "attn1.to_k.weight", "attn1.to_k.bias",
                  "attn1.to_v.weight", "attn1.to_v.bias",
                  "attn1.add_q_proj.weight", "attn1.add_q_proj.bias",
                  "attn1.add_k_proj.weight", "attn1.add_k_proj.bias",
                  "attn1.add_v_proj.weight", "attn1.add_v_proj.bias",
                  "net.0.proj.weight", "net.0.proj.bias",
                  "linear.weight", "linear.bias"]
    suffixes_2 = ["attn1.to_out.0.weight", "attn1.to_add_out.weight", "net.2.weight"]
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
    print(f"Missing keys: {missing_keys}, and")
    print(f"Unexpected keys: {unexpected_keys}, that's all.")
    del ckpt_dict