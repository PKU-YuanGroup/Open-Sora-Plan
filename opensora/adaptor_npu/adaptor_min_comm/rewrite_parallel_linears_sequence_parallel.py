import torch
from .min_comm_cfg import min_comm_config
from .cc_utils import set_context, async_gather_along_first_dim, reshape_to_2D, is_grad_needed


class RewriteColumnSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * min_comm_config.tp_world_size

        all_gather_buffer = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._all_gather_base(all_gather_buffer, input_, group=min_comm_config.tp_group)
        total_input = all_gather_buffer

        output_parallel = torch.matmul(total_input, weight.t())
        if bias is not None:
            output_parallel = output_parallel + bias
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)
        tp_group = min_comm_config.tp_group
        if is_grad_weight_needed:
            handle_all_gather, total_input = async_gather_along_first_dim(input_, tp_group,
                                                                          min_comm_config.tp_world_size)
            grad_input = grad_output.matmul(weight)
            handle_all_gather.wait()
            dim_size = list(input_.size())
            sub_grad_input = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle_reduce_scatter = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input, group=tp_group,
                                                                           async_op=True)
            grad_output = reshape_to_2D(grad_output)
            grad_weight = grad_output.t().matmul(reshape_to_2D(total_input))
            handle_reduce_scatter.wait()
            grad_bias = grad_output.sum(dim=0) if is_grad_bias_needed and ctx.use_bias else None
        else:
            grad_input = grad_output.matmul(weight)
            dim_size = list(input_.size())
            sub_grad_input = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle_reduce_scatter = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input, group=tp_group,
                                                                           async_op=True)
            handle_reduce_scatter.wait()
            grad_weight, grad_bias = None, None
        return grad_input, grad_weight, grad_bias


class RewriteRowSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        # ctx.world_size is needed for the case: rewrite forward (manually skipped) with cc backward
        ctx.world_size = min_comm_config.tp_world_size
        output_ = torch.matmul(input_, weight.t())
        if bias is not None:
            output_ = output_ + bias
        output_parallel = min_comm_config.reduce_scatter_along_first_dim(output_)
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors
        grad_output = min_comm_config.gather_along_first_dim(grad_output)
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
