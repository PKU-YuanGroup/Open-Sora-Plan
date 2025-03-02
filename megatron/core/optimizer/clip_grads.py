# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Gradient clipping."""

import os
from typing import List, Optional, Union

import amp_C
import torch
from apex.multi_tensor_apply import multi_tensor_applier
from torch import inf

from ..tensor_parallel import param_is_not_tensor_parallel_duplicate
from ..transformer.module import param_is_not_shared
from logging import getLogger
from megatron.training.global_vars import get_wandb_writer
import time

logger = getLogger()

class AdaptiveGradClipInfo:
    weight_norm = -1.0
    moving_avg_max_grad_norm = -1e6
    moving_avg_max_grad_norm_var = 0.0
    max_grad_norm = 0.0
    max_grad_norm_after_clip = 0.0
    max_norm = 0.0
    max_grad_norm_var = 0.0
    num_zero_grad = 0.0
    clip_coef = 1.0
    zero_grad_flag_list = None
    nan_norm_flag = 0


def get_adaptive_grad_clip_info(key):
    if hasattr(AdaptiveGradClipInfo, key):
        return getattr(AdaptiveGradClipInfo, key)
    else:
        logger.warning(f"AdaptiveGradClipInfo includes moving_avg_max_grad_norm,"
                "moving_avg_max_grad_norm_var, max_grad_norm, max_grad_norm_after_clip, max_norm," 
                "max_grad_norm_var, num_zero_grad, clip_coef, zero_grad_flag_list, nan_norm_flag,"
                "but {key} is not in the list.")
    return None


def gather_info_from_all_processes(info: Union[float, torch.Tensor], dtype=torch.int):
    if not torch.distributed.is_initialized():
        raise ValueError("torch.distributed is not initialized.")
    if not isinstance(info, torch.Tensor):
        info = torch.tensor([info], dtype=dtype).cuda()
    gathered_infos = [torch.zeros_like(info) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered_infos, info)
    if info.ndim == 0:
        gathered_infos = torch.stack(gathered_infos)
    else:
        gathered_infos = torch.cat(gathered_infos)
    return gathered_infos


def get_unlocked_params_weight_norm_fp32(params_for_norm, norm_type=2.0, model_parallel_group=None):
    # Calculate norm.
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in params_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if params_for_norm:
                weight_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [params_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                weight_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = weight_norm ** norm_type

        else:
            for p in params_for_norm:
                weight_norm = torch.norm(p.data, norm_type)
                total_norm += weight_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm ** (1.0 / norm_type)

    return total_norm

def zero_and_clip_grad_(grads, clip_coef=1.0, zero_grad_flag=True):
    if zero_grad_flag:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], 0
        )
    elif clip_coef != 1.0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], 1 / (clip_coef + 1.0e-6)
        )

def get_grad_norm(grads_for_norm, norm_type=2.0, model_parallel_group=None):
    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm ** (1.0 / norm_type)

    return total_norm

def adaptive_clip_grad_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    params_for_norm: Union[List[torch.Tensor], torch.Tensor] = None,
    norm_type: Union[int, float] = 2,
    clip_grad_ema_decay: float = 0.99,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized.
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(param.grad.detach())

    # Norm parameters.
    norm_type = float(norm_type)
    
    weight_norm = get_unlocked_params_weight_norm_fp32(params_for_norm, norm_type, model_parallel_group)

    grad_norm_before_clip = get_grad_norm(grads_for_norm, norm_type, model_parallel_group)
    grad_norm_before_clip_list = gather_info_from_all_processes(grad_norm_before_clip, dtype=torch.float)
    print(f"grad_norm_before_clip: {grad_norm_before_clip}, gathered_grad_norm_before_clip: {grad_norm_before_clip_list}")

    nan_norm_flag = 0
    if torch.isnan(grad_norm_before_clip_list).any() or torch.isinf(grad_norm_before_clip_list).any():
        nan_norm_flag = 1
        print(grad_norm_before_clip_list)
    nan_or_inf_list = torch.isnan(grad_norm_before_clip_list) | torch.isinf(grad_norm_before_clip_list)
    grad_norm_before_clip_list = torch.where(
        nan_or_inf_list, 
        torch.zeros_like(grad_norm_before_clip_list, device=grad_norm_before_clip_list.device), 
        grad_norm_before_clip_list
    )  # filter normal grad_norm
    max_grad_norm = grad_norm_before_clip_list.max().item()  # (rank, )
    moving_avg_max_grad_norm = AdaptiveGradClipInfo.moving_avg_max_grad_norm
    moving_avg_max_grad_norm_var = AdaptiveGradClipInfo.moving_avg_max_grad_norm_var
    ema_decay = clip_grad_ema_decay
    is_first_step = True if moving_avg_max_grad_norm < 0.0 else False # the value of init is -1e6, before first step

    if is_first_step:  
        moving_avg_max_grad_norm = 1.0
        moving_avg_max_grad_norm_var = 0.0
        max_grad_norm_var = 0.0
        max_norm = 1.0
        num_zero_grad = 0.0
        clip_coef = 1.0
        zero_grad_flag_list = torch.zeros_like(grad_norm_before_clip_list, device=grad_norm_before_clip_list.device)
        max_grad_norm_after_clip = 1.0  
        grad_norm_after_clip = grad_norm_before_clip
    else:
        # out of 3 sigma mean abnormal step.
        max_norm = moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)
        zero_grad_flag = torch.isnan(grad_norm_before_clip).any() or torch.isinf(grad_norm_before_clip).any() or grad_norm_before_clip > max_norm
        print(f"zero_grad_flag: {zero_grad_flag}")
        zero_grad_flag_list = gather_info_from_all_processes(zero_grad_flag, dtype=torch.int)
        print(f"zero_grad_flag_list: {zero_grad_flag_list}")
        clip_coef = torch.mean((~zero_grad_flag_list).float(), dim=-1, keepdim=True)
        zero_and_clip_grad_(grads, clip_coef=clip_coef, zero_grad_flag=zero_grad_flag)
        num_zero_grad = zero_grad_flag_list.sum().item()
        grad_norm_after_clip = get_grad_norm(grads_for_norm, norm_type, model_parallel_group)
        grad_norm_after_clip_list = gather_info_from_all_processes(grad_norm_after_clip, dtype=torch.float)
        max_grad_norm_after_clip = grad_norm_after_clip_list.max().item() * clip_coef
        # only communication bug can cause this situation
        if torch.isnan(grad_norm_after_clip_list).any() or torch.isinf(grad_norm_after_clip_list).any():
            print(grad_norm_after_clip_list)
            raise ValueError("Detected NaN or Inf in gathered clipping gradient norms.")
        # 用裁过的max grad norm作为ema统计量，裁过的一定是之前认为合理的最大范围
        if num_zero_grad < 2:
            moving_avg_max_grad_norm = ema_decay * moving_avg_max_grad_norm + (1 - ema_decay) * max_grad_norm_after_clip
            max_grad_norm_var = (moving_avg_max_grad_norm - max_grad_norm_after_clip) ** 2
            moving_avg_max_grad_norm_var = ema_decay * moving_avg_max_grad_norm_var + (1 - ema_decay) * max_grad_norm_var
        else:
            max_grad_norm_var = (moving_avg_max_grad_norm - max_grad_norm) ** 2

    AdaptiveGradClipInfo.weight_norm = weight_norm
    AdaptiveGradClipInfo.moving_avg_max_grad_norm = moving_avg_max_grad_norm
    AdaptiveGradClipInfo.moving_avg_max_grad_norm_var = moving_avg_max_grad_norm_var
    AdaptiveGradClipInfo.max_grad_norm = max_grad_norm
    AdaptiveGradClipInfo.max_grad_norm_after_clip = max_grad_norm_after_clip
    AdaptiveGradClipInfo.max_norm = max_norm
    AdaptiveGradClipInfo.max_grad_norm_var = max_grad_norm_var
    AdaptiveGradClipInfo.num_zero_grad = num_zero_grad
    AdaptiveGradClipInfo.clip_coef = clip_coef
    AdaptiveGradClipInfo.zero_grad_flag_list = zero_grad_flag_list
    AdaptiveGradClipInfo.nan_norm_flag = nan_norm_flag

    if torch.distributed.get_rank() == 0:
        wandb_writer = get_wandb_writer()
        if wandb_writer is not None:
            wandb_writer.log({
                'weight_norm': weight_norm,
                'moving_avg_max_grad_norm': AdaptiveGradClipInfo.moving_avg_max_grad_norm,
                'moving_avg_max_grad_norm_var': AdaptiveGradClipInfo.moving_avg_max_grad_norm_var,
                'max_grad_norm': AdaptiveGradClipInfo.max_grad_norm,
                'max_grad_norm_after_clip': AdaptiveGradClipInfo.max_grad_norm_after_clip,
                'max_norm': AdaptiveGradClipInfo.max_norm,
                'max_grad_norm_var': AdaptiveGradClipInfo.max_grad_norm_var,
                'num_zero_grad': AdaptiveGradClipInfo.num_zero_grad,
                'clip_coef': AdaptiveGradClipInfo.clip_coef,
                'nan_norm_flag': AdaptiveGradClipInfo.nan_norm_flag,
            }, commit=False)

    return grad_norm_after_clip

def clip_grad_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    norm_type: Union[int, float] = 2,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized.
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff
        )

    return total_norm


def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    model_parallel_group: torch.distributed.ProcessGroup,
) -> float:
    """Counts the number of zeros in gradients associated with the passed-in list of
    parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have the number of zeros in its corresponding
            gradient counted.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.tensor([0.0], dtype=torch.float, device='cuda')
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
    )

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros