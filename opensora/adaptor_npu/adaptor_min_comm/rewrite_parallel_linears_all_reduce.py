import torch
from .min_comm_cfg import min_comm_config
from .cc_utils import set_context, reshape_to_2D, is_grad_needed


class RewriteColumnAllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        output_parallel = torch.matmul(input_, weight.t())
        if bias is not None:
            output_parallel = output_parallel + bias
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        if is_grad_weight_needed:
            grad_input = grad_output.matmul(weight)
            handle = torch.distributed.all_reduce(grad_input, group=min_comm_config.tp_group, async_op=True)
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(input_))
            handle.wait()
            grad_bias = grad_output.sum(dim=0) if is_grad_bias_needed and ctx.use_bias else None
        else:
            grad_input = grad_output.matmul(weight)
            handle = torch.distributed.all_reduce(grad_input, group=min_comm_config.tp_group, async_op=True)
            handle.wait()
            grad_weight, grad_bias = None, None
        return grad_input, grad_weight, grad_bias


class RewriteRowAllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        output_ = torch.matmul(input_, weight.t())
        if bias is not None:
            output_ = output_ + bias
        output_parallel = min_comm_config.all_reduce(output_)
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        if is_grad_weight_needed:
            grad_input = grad_output.matmul(weight)
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(total_input))
            grad_bias = grad_output.sum(dim=0) if is_grad_bias_needed and ctx.use_bias else None
        else:
            grad_input = grad_output.matmul(weight)
            grad_weight, grad_bias = None, None
        return grad_input, grad_weight, grad_bias
