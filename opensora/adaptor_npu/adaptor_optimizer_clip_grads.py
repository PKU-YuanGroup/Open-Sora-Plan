import os
import sys
import math
import torch
import opensora.optimizer


def clip_grad_norm_fp32(parameters, grads_for_norm, max_norm, norm_type=2, model_parallel_group=None):
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
    if norm_type == math.inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group)
        total_norm = total_norm_cuda[0].item()
    else:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if not grads_for_norm:
            grad_norm = torch.cuda.FloatTensor([0])
            total_norm = grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coeff)
    return total_norm


for k, v in sys.modules.items():
    if 'opensora' in k and hasattr(v, 'clip_grad_norm_fp32'):
        setattr(v, 'clip_grad_norm_fp32', clip_grad_norm_fp32)
