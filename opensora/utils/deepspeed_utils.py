from deepspeed.utils import logger, instrument_w_nvtx
from deepspeed.runtime.utils import see_memory_usage, DummyOptim, get_weight_norm, clip_grad_norm_
from collections.abc import Iterable
from deepspeed.moe.utils import is_moe_param
import os
import psutil
import gc
from math import sqrt
from packaging import version as pkg_version

import torch
from deepspeed import comm as dist

try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf

from deepspeed.utils import groups, logger
from deepspeed.runtime.constants import PIPE_REPLICATED
from numpy import prod
from deepspeed.accelerator import get_accelerator

from deepspeed.module_inject.policy import transpose
from torch.nn import functional as F

from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank, is_model_parallel_parameter



def clip_grad_norm_(parameters, max_norm, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    all_norms = []
    if norm_type == inf:
        for p in parameters:
            all_norms.append(p.grad.data.abs().max().float())
        total_norm = torch.stack(all_norms).max()
        origin_device = total_norm.device.type
        total_norm = total_norm.to(get_accelerator().device_name())
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
    else:
        total_norm = 0
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank() == 0) or is_model_parallel_parameter(p):
                    param_norm = p.grad.data.detach().float().norm(norm_type)
                    all_norms.append(param_norm)
            else:
                param_norm = p.grad.data.detach().float().norm(norm_type)
                all_norms.append(param_norm)
        if len(all_norms) > 0:
            total_norm = torch.stack(all_norms).square().sum().float()
        else:
            total_norm = torch.FloatTensor([0.0]).to(parameters[0].device)
        origin_device = total_norm.device.type
        total_norm = total_norm.to(get_accelerator().device_name())
        # Sum across all model parallel GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm.pow(1. / norm_type)

    # Need to average total_norm across different GPUs due to the presence of moe params
    pg = groups._get_data_parallel_group()
    scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))
    scaled_norm_tensor = scaled_norm

    dist.all_reduce(scaled_norm_tensor, group=pg)
    total_norm = scaled_norm_tensor
    total_norm = total_norm.to(origin_device)

    max_norm = torch.tensor([float(max_norm)], device=parameters[0].device)
    clip_coef = max_norm / (total_norm + 1e-6)
    tmp_tensor = torch.tensor([1.0], device=parameters[0].device)
    clip_coef = torch.min(tmp_tensor, clip_coef)
    for p in parameters:
        p.grad.data.mul_(clip_coef)
    return total_norm, clip_coef


def get_grad_norm(parameters, norm_type=2, mpu=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be `'inf' for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
            
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.grad.data.float().norm(norm_type)
            total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        
        # Convert the sum back into the correct norm value
        total_norm_cuda = total_norm_cuda ** (1.0 / norm_type)

    # Handle inf, -inf, or NaN values.
    if total_norm_cuda.item() == float('inf') or total_norm_cuda.item() == -float('inf') or total_norm_cuda.item() != total_norm_cuda.item():
        total_norm_cuda = torch.tensor(-1.0, device=total_norm_cuda.device)

    return total_norm_cuda  


@instrument_w_nvtx
def backward(
    self, loss, allreduce_gradients=True, release_loss=False, retain_graph=False, scale_wrt_gas=True, 
    process_index=0, step_=0, moving_avg_grad_norm=-1e-6, moving_avg_grad_norm_std=0.0, accelerator=None, 
    ema_decay_grad_clipping=0.9999, 
    ):
    r"""Execute backward pass on the loss
    Arguments:
        loss: Torch tensor on which to execute backward propagation
        allreduce_gradients: is deprecated, ignored, and will soon be removed'
        retain_graph: bool, default: false
            forward on user defined choice of retain_graph
    """


    see_memory_usage("Engine before backward", force=self.memory_breakdown())

    if self.scale_wrt_gas is not None:
        scale_wrt_gas = self.scale_wrt_gas

    if not allreduce_gradients:
        logger.warning(f"Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed")

    # scale loss w.r.t. gradient accumulation if needed
    if self.gradient_accumulation_steps() > 1 and scale_wrt_gas:
        loss = self._scale_loss_by_gas(loss.float())

    # Log training loss
    mean_loss = loss.mean().detach()
    self.losses = mean_loss if self.losses is None else self.losses + mean_loss
    if self.monitor.enabled:
        if self.is_gradient_accumulation_boundary():
            if self.global_rank == 0:
                self.summary_events = [(
                    f"Train/Samples/train_loss",
                    self.losses.item(),
                    self.global_samples,
                )]
                self.monitor.write_events(self.summary_events)

    self._start_timers(self.engine_timers.backward_timers)

    assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
        "must provide optimizer during init in order to use backward"

    self._start_timers(self.engine_timers.backward_inner_timers)

    if self.zero_optimization():
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        self.optimizer.backward(loss, retain_graph=retain_graph)
    elif self.amp_enabled():
        # AMP requires delaying unscale when inside gradient accumulation boundaries
        # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
        delay_unscale = not self.is_gradient_accumulation_boundary()
        with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    elif self.fp16_enabled():
        if self.eigenvalue_enabled():
            self.optimizer.backward(loss, create_graph=True, retain_graph=True)
        else:
            self.optimizer.backward(loss, retain_graph=retain_graph)
    elif self.bfloat16_enabled():
        self.optimizer.backward(loss)
    else:
        if self.eigenvalue_enabled():
            loss.backward(create_graph=True, retain_graph=True)
        else:
            loss.backward(retain_graph=retain_graph)

    # ==============================================
    weight_norm = get_weight_norm(parameters=self.module.parameters(), mpu=self.mpu)

    grad_norm = get_grad_norm(parameters=self.module.parameters(), mpu=self.mpu)
    grad_norm_list = accelerator.gather(grad_norm)  # (rank, )
    grad_norm = grad_norm_list.mean().item() 
    grad_norm_std = grad_norm_list.std().item()

    is_first_step = True if moving_avg_grad_norm < 0.0 else False # the value of init is -1e6, before first step
    ema_decay = ema_decay_grad_clipping
    if is_first_step:  
        moving_avg_grad_norm = grad_norm
        moving_avg_grad_norm_std = grad_norm_std
    else:  # other steps use ema_decay
        moving_avg_grad_norm = ema_decay * moving_avg_grad_norm + (1 - ema_decay) * grad_norm
        moving_avg_grad_norm_std = ema_decay * moving_avg_grad_norm_std + (1 - ema_decay) * grad_norm_std

    max_norm = min(moving_avg_grad_norm + 3.0 * moving_avg_grad_norm_std, self.gradient_clipping())
    _, clip_coef = clip_grad_norm_(parameters=self.module.parameters(), max_norm=max_norm, mpu=self.mpu)
    grad_norm_clip = get_grad_norm(parameters=self.module.parameters(), mpu=self.mpu)
    grad_norm_clip = accelerator.gather(grad_norm_clip).mean().item()
    
    if clip_coef != 1.0:
        print(f'rank {process_index} | step {step_} | after gather grad_norm {grad_norm} | after clip_grad_norm_ {grad_norm_clip}')
    # =======================================================================================

    self._stop_timers(self.engine_timers.backward_inner_timers)

    self._start_timers(self.engine_timers.backward_reduce_timers)
    if allreduce_gradients and self.enable_backward_allreduce:
        # Traditional code path that allreduces the module parameter grads
        self.allreduce_gradients()

    self._stop_timers(self.engine_timers.backward_reduce_timers)

    self._stop_timers(self.engine_timers.backward_timers)

    if release_loss:
        # loss.data = None
        pass

    see_memory_usage("Engine after backward", force=self.memory_breakdown())

    return loss, grad_norm, weight_norm, moving_avg_grad_norm, grad_norm_clip, max_norm, moving_avg_grad_norm_std, grad_norm_std, clip_coef