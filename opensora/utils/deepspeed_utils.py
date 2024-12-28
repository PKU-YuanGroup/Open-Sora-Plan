from deepspeed.utils import logger, instrument_w_nvtx
from deepspeed.runtime.utils import see_memory_usage, DummyOptim, get_weight_norm, clip_grad_norm_
from collections.abc import Iterable
from deepspeed.moe.utils import is_moe_param
import os
import psutil
import gc
from math import sqrt
from packaging import version as pkg_version
import accelerate
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
from accelerate.state import AcceleratorState
from deepspeed.module_inject.policy import transpose
from torch.nn import functional as F

from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank, is_model_parallel_parameter

def safe_get_weight_norm(module):
    if module is None:
        return None
    else:
        return get_weight_norm(parameters=module.parameters(), mpu=None)

def get_weight_norm_dict(model):
    weight_norm_dict = {}
    cnt = 0
    for m in model.transformer_blocks:
        for sub_m in m:
            weight_norm_dict.update({f'weight_norm_block/block_{cnt}': safe_get_weight_norm(sub_m)})
            cnt += 1
    weight_norm_dict.update({f'weight_norm_others/norm_final': safe_get_weight_norm(model.norm_final)})
    weight_norm_dict.update({f'weight_norm_others/proj_out': safe_get_weight_norm(model.proj_out)})
    weight_norm_dict.update({f'weight_norm_others/patch_embed': safe_get_weight_norm(model.patch_embed)})
    weight_norm_dict.update({f'weight_norm_others/time_text_embed': safe_get_weight_norm(model.time_text_embed)})
    weight_norm_dict.update({f'weight_norm_others/caption_projection': safe_get_weight_norm(model.caption_projection)})
    if getattr(model, 'weight_norm_skip/skip_norm_linear', None) is not None:
        wn = safe_get_weight_norm(model.skip_norm_linear[0])
        if wn is not None and wn > 0.0:
            weight_norm_dict.update({f'skip_{i}': safe_get_weight_norm(m) for i, m in enumerate(model.skip_norm_linear)})
    if getattr(model, 'weight_norm_others/final_conv', None) is not None:
        wn = safe_get_weight_norm(model.final_conv)
        if wn is not None and wn > 0.0:
            weight_norm_dict.update({f'final_conv': wn})
    if getattr(model, 'weight_norm_others/caption_projection_2', None) is not None:
        wn = safe_get_weight_norm(model.caption_projection_2)
        if wn is not None and wn > 0.0:
            weight_norm_dict.update({f'weight_norm_others/caption_projection_2': wn})
    return weight_norm_dict
def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

def get_weight_norm_for_ema(parameters, norm_type=2):
    """Get norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
        -1 if the norm value is NaN or Inf.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    norm_type = float(norm_type)
    
    total_norm = 0.
    for p in parameters:

        param_norm = p.data.float().norm(norm_type)
        total_norm += param_norm**norm_type

    # Sum across all model parallel GPUs.
    total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
    total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

# def clip_grad_norm_(parameters, max_norm, norm_type=2, mpu=None, clip=True, accelerator=None):
#     """Clips gradient norm of an iterable of parameters.

#     This has been adapted from Nvidia megatron. We add norm averaging
#     to consider MoE params when calculating norm as they will result
#     in different norms across different ranks.

#     This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
#     added functionality to handle model parallel parameters. Note that
#     the gradients are modified in place.

#     Arguments:
#         parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
#             single Tensor that will have gradients normalized
#         max_norm (float or int): max norm of the gradients
#         norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
#             infinity norm.

#     Returns:
#         Total norm of the parameters (viewed as a single vector).
#     """
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     norm_type = float(norm_type)
#     all_norms = []
#     if norm_type == inf:
#         for p in parameters:
#             all_norms.append(p.grad.data.abs().max().float())
#         total_norm = torch.stack(all_norms).max()
#         origin_device = total_norm.device.type
#         total_norm = total_norm.to(get_accelerator().device_name())
#         # Take max across all GPUs.
#         if mpu is not None:
#             dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
#     else:
#         total_norm = 0
#         for p in parameters:
#             if mpu is not None:
#                 if (mpu.get_model_parallel_rank() == 0) or is_model_parallel_parameter(p):
#                     param_norm = p.grad.data.detach().float().norm(norm_type)
#                     all_norms.append(param_norm)
#             else:
#                 param_norm = p.grad.data.detach().float().norm(norm_type)
#                 all_norms.append(param_norm)
#         if len(all_norms) > 0:
#             total_norm = torch.stack(all_norms).square().sum().float()
#         else:
#             total_norm = torch.FloatTensor([0.0]).to(parameters[0].device)
#         origin_device = total_norm.device.type
#         total_norm = total_norm.to(get_accelerator().device_name())
#         # Sum across all model parallel GPUs.
#         if mpu is not None:
#             dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
#         total_norm = total_norm.pow(1. / norm_type)

#     if not clip:
#         return total_norm
#     # Need to average total_norm across different GPUs due to the presence of moe params
#     # pg = groups._get_data_parallel_group()
#     # print('befor scale', dist.get_world_size(group=pg), total_norm, flush=True)
#     # scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))
#     # scaled_norm_tensor = scaled_norm

#     # dist.all_reduce(scaled_norm_tensor, group=pg)
#     # total_norm = scaled_norm_tensor
#     # total_norm = total_norm.to(origin_device)
#     # print('after scale', total_norm, flush=True)
#     need_zeros_ = torch.isnan(total_norm).any() or torch.isinf(total_norm).any()  # nan or inf in abnormal local rank
#     total_norm_list = accelerator.gather(total_norm)
#     nan_or_inf_list = torch.isnan(total_norm_list) | torch.isinf(total_norm_list)
#     total_norm_list = torch.where(
#         nan_or_inf_list, 
#         torch.zeros_like(total_norm_list, device=total_norm_list.device), 
#         total_norm_list
#         )  # filter normal grad_norm to calculate clip_coef for other normal ranks
#     total_norm = total_norm_list.max()

#     max_norm = torch.tensor([float(max_norm)], device=parameters[0].device)
#     clip_coef = max_norm / (total_norm + 1e-6)
#     tmp_tensor = torch.tensor([1.0], device=parameters[0].device)
#     clip_coef = torch.min(tmp_tensor, clip_coef)
#     for p in parameters:
#         if need_zeros_:  # zero_ in abnormal local rank
#             p.grad.data.zero_()
#         else:  # clipping in other normal ranks
#             p.grad.data.mul_(clip_coef)
#     return total_norm, clip_coef





# @instrument_w_nvtx
# def backward(
#     self, loss, allreduce_gradients=True, release_loss=False, retain_graph=False, scale_wrt_gas=True, 
#     process_index=0, step_=0, moving_avg_max_grad_norm=-1e-6, moving_avg_max_grad_norm_var=0.0, accelerator=None, 
#     ema_decay_grad_clipping=0.9999
#     ):
#     r"""Execute backward pass on the loss
#     Arguments:
#         loss: Torch tensor on which to execute backward propagation
#         allreduce_gradients: is deprecated, ignored, and will soon be removed'
#         retain_graph: bool, default: false
#             forward on user defined choice of retain_graph
#     """


#     see_memory_usage("Engine before backward", force=self.memory_breakdown())

#     if self.scale_wrt_gas is not None:
#         scale_wrt_gas = self.scale_wrt_gas

#     if not allreduce_gradients:
#         logger.warning(f"Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed")

#     # scale loss w.r.t. gradient accumulation if needed
#     if self.gradient_accumulation_steps() > 1 and scale_wrt_gas:
#         loss = self._scale_loss_by_gas(loss.float())

#     # Log training loss
#     mean_loss = loss.mean().detach()
#     self.losses = mean_loss if self.losses is None else self.losses + mean_loss
#     if self.monitor.enabled:
#         if self.is_gradient_accumulation_boundary():
#             if self.global_rank == 0:
#                 self.summary_events = [(
#                     f"Train/Samples/train_loss",
#                     self.losses.item(),
#                     self.global_samples,
#                 )]
#                 self.monitor.write_events(self.summary_events)

#     self._start_timers(self.engine_timers.backward_timers)

#     assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
#         "must provide optimizer during init in order to use backward"

#     self._start_timers(self.engine_timers.backward_inner_timers)

#     if self.zero_optimization():
#         self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
#         self.optimizer.backward(loss, retain_graph=retain_graph)
#     elif self.amp_enabled():
#         # AMP requires delaying unscale when inside gradient accumulation boundaries
#         # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
#         delay_unscale = not self.is_gradient_accumulation_boundary()
#         with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
#             scaled_loss.backward(retain_graph=retain_graph)
#     elif self.fp16_enabled():
#         if self.eigenvalue_enabled():
#             self.optimizer.backward(loss, create_graph=True, retain_graph=True)
#         else:
#             self.optimizer.backward(loss, retain_graph=retain_graph)
#     elif self.bfloat16_enabled():
#         self.optimizer.backward(loss)
#     else:
#         if self.eigenvalue_enabled():
#             loss.backward(create_graph=True, retain_graph=True)
#         else:
#             loss.backward(retain_graph=retain_graph)

#     # ==============================================
#     weight_norm = get_weight_norm(parameters=self.module.parameters(), mpu=self.mpu)

#     grad_norm = clip_grad_norm_(parameters=self.module.parameters(), max_norm=None, mpu=self.mpu, clip=False)
#     grad_norm_list = accelerator.gather(grad_norm)

#     detect_nan = 0
#     if torch.isnan(grad_norm_list).any() or torch.isinf(grad_norm_list).any():
#         detect_nan = 1
#         print(grad_norm_list)

#     nan_or_inf_list = torch.isnan(grad_norm_list) | torch.isinf(grad_norm_list)
#     grad_norm_list = torch.where(
#         nan_or_inf_list, 
#         torch.zeros_like(grad_norm_list, device=grad_norm_list.device), 
#         grad_norm_list
#         )  # filter normal grad_norm
#     # print(f"process_index {process_index}, step_ {step_}, grad_norm_list {grad_norm_list}")

#     max_grad_norm = grad_norm_list.max().item()  # (rank, )

#     is_first_step = True if moving_avg_max_grad_norm < 0.0 else False # the value of init is -1e6, before first step
#     ema_decay = ema_decay_grad_clipping

#     if is_first_step:  
#         moving_avg_max_grad_norm = max_grad_norm
#         moving_avg_max_grad_norm_var = 0
#         max_grad_norm_var = max_grad_norm
#         max_norm = 1.0
#         clip_coef = 1.0
#         max_grad_norm_clip = max_grad_norm
#     else:
#         # out of 1 sigma mean abnormal step.
#         max_norm = min(moving_avg_max_grad_norm + 2.0 * (moving_avg_max_grad_norm_var ** 0.5), self.gradient_clipping())
#         _, clip_coef = clip_grad_norm_(parameters=self.module.parameters(), max_norm=max_norm, mpu=self.mpu, accelerator=accelerator)
#         grad_norm_clip = clip_grad_norm_(parameters=self.module.parameters(), max_norm=None, mpu=self.mpu, clip=False)
#         grad_norm_clip_list = accelerator.gather(grad_norm_clip)
#         # print(f"process_index {process_index}, step_ {step_}, grad_norm_clip_list {grad_norm_clip_list}")
#         max_grad_norm_clip = grad_norm_clip_list.max().item()
#         if torch.isnan(grad_norm_clip_list).any() or torch.isinf(grad_norm_clip_list).any():
#             print(grad_norm_clip_list)
#             raise ValueError("Detected NaN or Inf in gathered clipping gradient norms.")
#         if clip_coef == 1.0:  # mean normal step!!! otherwise we do not update ema.
#             moving_avg_max_grad_norm = ema_decay * moving_avg_max_grad_norm + (1 - ema_decay) * max_grad_norm
#             max_grad_norm_var = (moving_avg_max_grad_norm - max_grad_norm) ** 2
#             moving_avg_max_grad_norm_var = ema_decay * moving_avg_max_grad_norm_var + (1 - ema_decay) * max_grad_norm_var
#         else:
#             max_grad_norm_var = (moving_avg_max_grad_norm - max_grad_norm) ** 2
#     # =======================================================================================

#     self._stop_timers(self.engine_timers.backward_inner_timers)

#     self._start_timers(self.engine_timers.backward_reduce_timers)
#     if allreduce_gradients and self.enable_backward_allreduce:
#         # Traditional code path that allreduces the module parameter grads
#         self.allreduce_gradients()

#     self._stop_timers(self.engine_timers.backward_reduce_timers)

#     self._stop_timers(self.engine_timers.backward_timers)

#     if release_loss:
#         # loss.data = None
#         pass

#     see_memory_usage("Engine after backward", force=self.memory_breakdown())

#     return loss, max_grad_norm, weight_norm, moving_avg_max_grad_norm, max_grad_norm_clip, max_norm, moving_avg_max_grad_norm_var, max_grad_norm_var, clip_coef, detect_nan






def zero_grad_abnormal_(parameters, max_norm, norm_type=2, mpu=None, clip=True, accelerator=None, force_zero_grad_step=0, step_=0):
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
    # import ipdb;ipdb.set_trace()
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

    if not clip:
        return total_norm
    
    zero_grad = torch.isnan(total_norm).any() or torch.isinf(total_norm).any()  # nan or inf in abnormal local rank
    if not zero_grad:
        zero_grad = total_norm > max_norm

    zero_grad_list = accelerator.gather(zero_grad)
    clip_coef = torch.mean((~zero_grad_list).float(), dim=-1, keepdim=True)

    # -------------for debug all batch zero grad---------------
    force_zero_grad = force_zero_grad_step == step_
    if force_zero_grad:
        zero_grad_list = torch.ones_like(zero_grad_list, device=zero_grad_list.device)
        clip_coef = torch.FloatTensor([0.0]).to(parameters[0].device)
        for p in parameters:
            p.grad.data.mul_(clip_coef)
        return total_norm, zero_grad_list, clip_coef
    # -------------for debug all batch zero grad---------------
    
    if zero_grad:  # zero_ in abnormal local rank
        for p in parameters:
            p.grad.data.zero_()
    elif clip_coef != 1.0:
        for p in parameters:
            p.grad.data.div_(clip_coef)
    return total_norm, zero_grad_list, clip_coef


def safe_get_grad_norm(module, accelerator):
    # import ipdb;ipdb.set_trace()
    params = list(module.parameters())
    if len(params) == 0 or params[0].grad is None:
        return 0.0
    grad_norm = zero_grad_abnormal_(parameters=params, max_norm=None, mpu=None, clip=False)
    # return grad_norm.mean().item()
    grad_norm_list = accelerator.gather(grad_norm)
    return grad_norm_list.mean().item()
def get_grad_norm_dict(model, accelerator):

    grad_norm_dict = {}
    cnt = 0
    for m in model.transformer_blocks:
        for sub_m in m:
            grad_norm_dict.update({f'grad_norm_block/block_{cnt}': safe_get_grad_norm(sub_m, accelerator)})
            cnt += 1
    # import ipdb;ipdb.set_trace()
    grad_norm_dict.update({f'grad_norm_others/norm_final': safe_get_grad_norm(model.norm_final, accelerator)})
    grad_norm_dict.update({f'grad_norm_others/proj_out': safe_get_grad_norm(model.proj_out, accelerator)})
    grad_norm_dict.update({f'grad_norm_others/patch_embed': safe_get_grad_norm(model.patch_embed, accelerator)})
    grad_norm_dict.update({f'grad_norm_others/time_text_embed': safe_get_grad_norm(model.time_text_embed, accelerator)})
    grad_norm_dict.update({f'grad_norm_others/caption_projection': safe_get_grad_norm(model.caption_projection, accelerator)})
    if getattr(model, 'grad_norm_skip/skip_norm_linear', None) is not None:
        wn = safe_get_grad_norm(model.skip_norm_linear[0], accelerator)
        if wn is not None and wn > 0.0:
            grad_norm_dict.update({f'skip_{i}': safe_get_grad_norm(m, accelerator) for i, m in enumerate(model.skip_norm_linear)})
    if getattr(model, 'grad_normothers/final_conv', None) is not None:
        wn = safe_get_grad_norm(model.final_conv, accelerator)
        if wn is not None and wn > 0.0:
            grad_norm_dict.update({f'final_conv': wn})
    if getattr(model, 'grad_norm_others/caption_projection_2', None) is not None:
        wn = safe_get_grad_norm(model.caption_projection_2, accelerator)
        if wn is not None and wn > 0.0:
            grad_norm_dict.update({f'grad_norm_others/caption_projection_2': wn})
    return grad_norm_dict


@instrument_w_nvtx
def backward(
    self, loss, allreduce_gradients=True, release_loss=False, retain_graph=False, scale_wrt_gas=True, 
    step_=0, moving_avg_max_grad_norm=-1e-6, moving_avg_max_grad_norm_var=0.0, accelerator=None, 
    ema_decay_grad_clipping=0.99, force_zero_grad_step=-1, log_detail_norm_freq=10, 
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

    if step_ % log_detail_norm_freq == 0 and step_ >= log_detail_norm_freq:
        grad_norm_dict = get_grad_norm_dict(self.module, accelerator)
    else:
        grad_norm_dict = {}
    grad_norm = zero_grad_abnormal_(parameters=self.module.parameters(), max_norm=None, mpu=self.mpu, clip=False)
    grad_norm_list = accelerator.gather(grad_norm)

    detect_nan = 0
    if torch.isnan(grad_norm_list).any() or torch.isinf(grad_norm_list).any():
        detect_nan = 1
        print(grad_norm_list)

    nan_or_inf_list = torch.isnan(grad_norm_list) | torch.isinf(grad_norm_list)
    grad_norm_list = torch.where(
        nan_or_inf_list, 
        torch.zeros_like(grad_norm_list, device=grad_norm_list.device), 
        grad_norm_list
        )  # filter normal grad_norm

    max_grad_norm = grad_norm_list.max().item()  # (rank, )

    is_first_step = True if moving_avg_max_grad_norm < 0.0 else False # the value of init is -1e6, before first step
    ema_decay = ema_decay_grad_clipping

    if is_first_step:  
        moving_avg_max_grad_norm = max_grad_norm
        moving_avg_max_grad_norm_var = max_grad_norm
        max_grad_norm_var = max_grad_norm
        max_norm = 1.0
        num_zero_grad = 0.0
        clip_coef = 1.0
        zero_grad_list = torch.zeros_like(grad_norm_list, device=grad_norm_list.device)
        max_grad_norm_clip = max_grad_norm  
    else:
        # out of 3 sigma mean abnormal step.
        max_norm = moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)
        _, zero_grad_list, clip_coef = zero_grad_abnormal_(
            parameters=self.module.parameters(), max_norm=max_norm, mpu=self.mpu, accelerator=accelerator, step_=step_, force_zero_grad_step=force_zero_grad_step)
        num_zero_grad = zero_grad_list.sum().item()
        grad_norm_clip = zero_grad_abnormal_(parameters=self.module.parameters(), max_norm=None, mpu=self.mpu, clip=False, accelerator=accelerator)
        grad_norm_clip_list = accelerator.gather(grad_norm_clip)
        # print(f"process_index {process_index}, step_ {step_}, grad_norm_clip_list {grad_norm_clip_list}")
        max_grad_norm_clip = grad_norm_clip_list.max().item() * clip_coef
        if torch.isnan(grad_norm_clip_list).any() or torch.isinf(grad_norm_clip_list).any():
            print(grad_norm_clip_list)
            raise ValueError("Detected NaN or Inf in gathered clipping gradient norms.")
        # 用裁过的max grad norm作为ema统计量，裁过的一定是之前认为合理的最大范围
        if num_zero_grad < 2:
            moving_avg_max_grad_norm = ema_decay * moving_avg_max_grad_norm + (1 - ema_decay) * max_grad_norm_clip
            max_grad_norm_var = (moving_avg_max_grad_norm - max_grad_norm_clip) ** 2
            moving_avg_max_grad_norm_var = ema_decay * moving_avg_max_grad_norm_var + (1 - ema_decay) * max_grad_norm_var
        else:
            max_grad_norm_var = (moving_avg_max_grad_norm - max_grad_norm) ** 2
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

    return loss, max_grad_norm, weight_norm, moving_avg_max_grad_norm, max_grad_norm_clip, max_norm, \
        moving_avg_max_grad_norm_var, max_grad_norm_var, num_zero_grad, detect_nan, clip_coef, zero_grad_list, grad_norm_dict


