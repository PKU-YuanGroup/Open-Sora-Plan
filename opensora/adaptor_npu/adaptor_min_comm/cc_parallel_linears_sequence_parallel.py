import torch
from .min_comm_cfg import min_comm_config
from .matmul_soc_friendly import get_aligned_mm_inputs
from .cc_utils import CommunicationType, CCParallel
from .cc_utils import shuffle_as_cc_reduce_scatter, shuffle_as_cc_all_gather
from .cc_utils import set_context, reshape_to_2D, async_gather_along_first_dim, is_grad_needed, get_parallel_num
from .rewrite_parallel_linears_sequence_parallel import RewriteColumnSeqParallelFunction, RewriteRowSeqParallelFunction


class CCColumnSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)

        parallel_num = get_parallel_num(input_.shape[0] * input_.shape[1] * min_comm_config.tp_world_size,
                                        weight.shape[1], weight.shape[0])
        if parallel_num == 1:
            return RewriteColumnSeqParallelFunction.forward(ctx, input_, weight, bias)

        trans_weight = weight.t()

        if min_comm_config.matmul_soc_friendly_enabled:
            input_, trans_weight = get_aligned_mm_inputs(input_, trans_weight, sp_coef=min_comm_config.tp_world_size,
                                                         parallel_num=parallel_num)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, trans_weight, out=output_tensor)
            if bias is not None:
                output_tensor.add_(bias)
            return output_tensor

        cc_parallel = CCParallel(input_, CommunicationType.ALL_GATHER, compute_fcn, compute_first=False,
                                 weight_shape_list=list(trans_weight.shape), parallel_num=parallel_num)
        output = cc_parallel.run()
        output = shuffle_as_cc_reduce_scatter(output, min_comm_config.tp_world_size, parallel_num)
        if not min_comm_config.all_gather_recomputation_enabled:
            ctx.total_input = shuffle_as_cc_reduce_scatter(cc_parallel.comm_output, min_comm_config.tp_world_size,
                                                           parallel_num)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        grad_input_orig_shape = list(grad_output.shape[:-1]) + list([weight.shape[-1]])
        grad_output = reshape_to_2D(grad_output)
        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)

        if is_grad_weight_needed:
            if min_comm_config.all_gather_recomputation_enabled:
                total_input_work, total_input = async_gather_along_first_dim(input_, min_comm_config.tp_group,
                                                                             min_comm_config.tp_world_size)
            else:
                total_input = ctx.total_input

            # if grad_output.shape[-1] is not 512B aligned, transpose its memory alignment but keep its shape
            if grad_output.is_contiguous() and (grad_output.shape[-1] * grad_output.element_size()) % 512 > 0:
                grad_output = grad_output.t().contiguous().t()
            grad_input = grad_output.matmul(weight)
            grad_input = grad_input.reshape(grad_input_orig_shape)
            dim_size = list(input_.size())
            sub_grad_input = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
            sub_grad_input_work = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                                         group=min_comm_config.tp_group, async_op=True)
            if min_comm_config.all_gather_recomputation_enabled:
                total_input_work.wait()
            total_input = reshape_to_2D(total_input)
            grad_weight = grad_output.t().matmul(total_input)
            sub_grad_input_work.wait()
            if is_grad_bias_needed and ctx.use_bias:
                grad_bias = grad_output.sum(dim=0) if grad_output.is_contiguous() else grad_output.t().sum(dim=1)
            else:
                grad_bias = None
        else:
            # if grad_output.shape[-1] is not 512B aligned, transpose its memory alignment but keep its shape
            if grad_output.is_contiguous() and (grad_output.shape[-1] * grad_output.element_size()) % 512 > 0:
                grad_output = grad_output.t().contiguous().t()
            grad_input = grad_output.matmul(weight)
            grad_input = grad_input.reshape(grad_input_orig_shape)
            dim_size = list(input_.size())
            sub_grad_input = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
            torch.distributed._reduce_scatter_base(sub_grad_input, grad_input, group=min_comm_config.tp_group)
            grad_weight, grad_bias = None, None
        return sub_grad_input, grad_weight, grad_bias


class CCRowSeqParallelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias):
        set_context(ctx, input_, weight, bias)
        ctx.world_size = min_comm_config.tp_world_size

        parallel_num = get_parallel_num(input_.shape[0] * input_.shape[1], weight.shape[1], weight.shape[0])
        if parallel_num == 1:
            return RewriteRowSeqParallelFunction.forward(ctx, input_, weight, bias)

        trans_weight = weight.t()

        if min_comm_config.matmul_soc_friendly_enabled:
            input_, trans_weight = get_aligned_mm_inputs(input_, trans_weight, parallel_num=parallel_num)

        def compute_fcn(input_tensor):
            sub_output = torch.matmul(input_tensor, trans_weight)
            if bias is not None:
                sub_output = sub_output + bias
            return sub_output

        input_ = shuffle_as_cc_all_gather(input_, ctx.world_size, parallel_num)
        cc_reduce_scatter = CCParallel(input_, CommunicationType.REDUCE_SCATTER, compute_fcn, compute_first=True,
                                       weight_shape_list=list(trans_weight.shape), parallel_num=parallel_num)
        output_ = cc_reduce_scatter.run()
        return output_

    @staticmethod
    def backward(ctx, grad_output):
        total_input, weight = ctx.saved_tensors

        parallel_num = get_parallel_num(grad_output.shape[0] * grad_output.shape[1] * min_comm_config.tp_world_size,
                                        weight.shape[0], weight.shape[1])
        if parallel_num == 1:
            return RewriteRowSeqParallelFunction.backward(ctx, grad_output)

        if min_comm_config.matmul_soc_friendly_enabled:
            grad_output, weight = get_aligned_mm_inputs(grad_output, weight, sp_coef=min_comm_config.tp_world_size,
                                                        parallel_num=parallel_num)

        def compute_fcn(input_tensor, output_tensor):
            torch.matmul(input_tensor, weight, out=output_tensor)
            return output_tensor

        is_grad_weight_needed, is_grad_bias_needed = is_grad_needed(ctx.needs_input_grad)

        if is_grad_weight_needed:
            cc_all_gather = CCParallel(grad_output, CommunicationType.ALL_GATHER, compute_fcn, compute_first=False,
                                       weight_shape_list=list(weight.shape), parallel_num=parallel_num)
            grad_input = cc_all_gather.run()
            grad_input = shuffle_as_cc_reduce_scatter(grad_input, ctx.world_size, parallel_num)
            grad_output = reshape_to_2D(cc_all_gather.comm_output)
            grad_output = shuffle_as_cc_reduce_scatter(grad_output, ctx.world_size, parallel_num)
            total_input = reshape_to_2D(total_input)
            grad_weight = grad_output.t().matmul(total_input)
            if is_grad_bias_needed and ctx.use_bias:
                grad_bias = grad_output.sum(dim=0) if grad_output.is_contiguous() else grad_output.t().sum(dim=1)
            else:
                grad_bias = None
        else:
            cc_all_gather = CCParallel(grad_output, CommunicationType.ALL_GATHER, compute_fcn, compute_first=False,
                                       weight_shape_list=list(weight.shape), parallel_num=parallel_num)
            grad_input = cc_all_gather.run()
            grad_input = shuffle_as_cc_reduce_scatter(grad_input, ctx.world_size, parallel_num)
            grad_weight, grad_bias = None, None
        return grad_input, grad_weight, grad_bias
