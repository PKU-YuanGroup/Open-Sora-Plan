from functools import reduce
import math

import torch
import opensora.model
import opensora.optimizer
from opensora.core import mpu, tensor_parallel
from opensora.model.distributed import MemoryBuffer
from opensora.global_vars import get_args


def DistributedOptimizerInit(self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
                             use_contiguous_buffers_in_local_ddp, fp16, bf16, params_dtype, grad_scaler, models):
    super(opensora.optimizer.distrib_optimizer.DistributedOptimizer, self).__init__(
        optimizer, clip_grad, log_num_zeros_in_grad,
        params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        fp16, bf16, params_dtype, grad_scaler, models)

    # Verify that contiguous buffers are being used.
    # - Note: this should already be checked in arguments.py.
    assert use_contiguous_buffers_in_local_ddp

    # Model grad buffer ranges.
    self.model_gbuf_ranges = []
    for model_index, model in enumerate(self.models):
        self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model))
    self.model_param_gbuf_map = \
        self.build_model_param_gbuf_map(self.model_gbuf_ranges)

    # Optimizer ranges.
    self.opt_group_ranges = self.build_optimizer_group_ranges(
        self.optimizer.param_groups,
        self.model_gbuf_ranges)

    # Allocate main param shards.
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = self.build_model_and_main_param_groups(self.model_gbuf_ranges,
                                               self.model_param_gbuf_map,
                                               self.opt_group_ranges)

    # Initialize param buffers.
    # - These are views on the DDP model's grad buffers, that share
    #   storage & have their own dtype. This is safe because the param
    #   dtype size is always <= grad dtype size.
    self.param_buffers = []
    for model_index, model in enumerate(self.models):
        current_param_buffers = {}
        for dtype, grad_buffer in model._grad_buffers.items():
            # create NPU tensor with set_() instead of tensor.storage()._untyped()
            param_buffer = torch.tensor(torch.flatten(grad_buffer.data),  # grad_buffer.data.storage()._untyped(),
                                        dtype=params_dtype,
                                        device=grad_buffer.data.device)

            param_buffer = param_buffer[:grad_buffer.numel_padded]
            current_param_buffers[dtype] = param_buffer
        self.param_buffers.append(current_param_buffers)

    # Update optimizer groups.
    # - Also, leverage state_dict() and load_state_dict() to
    #   recast preexisting per-param state tensors.
    self.optimizer.param_groups = \
        [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())


def build_model_and_main_param_groups(cls,
                                      model_gbuf_ranges,
                                      param_gbuf_map,
                                      opt_group_ranges):
    """
    Create main parameter groups needed for the optimizer step.

    These groups encompass both: 1) groups used by this class, for
    reducing/gather, and 2) groups used by the inner optimizer for the
    parameter update. Given that the conceptual grad buffer partitioning
    (created in earlier method) doesn't respect parameter boundaries,
    the optimizer operates on shards of the model parameters, rather than
    the full parameters.
    """

    # Parameter groups:
    #   model_float16_groups: original float16 parameters
    #   model_fp32_groups: original fp32 parameters
    #   shard_float16_groups: shards of original float16 parameters
    #   shard_fp32_groups: shards of original fp32 parameters
    #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
    model_float16_groups = []
    model_fp32_groups = []
    shard_float16_groups = []
    shard_fp32_groups = []
    shard_fp32_from_float16_groups = []

    # Allocate (or slice) each group's param shard.
    for group_index, group_range in enumerate(opt_group_ranges):

        # Params of this group.
        model_float16_params_this_group = []
        model_fp32_params_this_group = []
        shard_float16_params_this_group = []
        shard_fp32_params_this_group = []
        shard_fp32_from_float16_params_this_group = []
        model_float16_groups.append(model_float16_params_this_group)
        model_fp32_groups.append(model_fp32_params_this_group)
        shard_float16_groups.append(shard_float16_params_this_group)
        shard_fp32_groups.append(shard_fp32_params_this_group)
        shard_fp32_from_float16_groups.append(
            shard_fp32_from_float16_params_this_group)

        for model_param in group_range["params"]:

            assert model_param.requires_grad

            model_index, dtype = param_gbuf_map[model_param]
            gbuf_range = model_gbuf_ranges[model_index][dtype]
            param_range = gbuf_range["param_map"][model_param]["param"]
            # fp16, bf16 params.
            if model_param.type() in ['torch.cuda.HalfTensor',
                                      'torch.cuda.BFloat16Tensor',
                                      'torch.npu.BFloat16Tensor']:

                # Clone model -> main.
                shard_model_param = model_param.detach().view(-1) \
                    [param_range.start:param_range.end]
                shard_main_param = shard_model_param.clone().float()
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_main_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared
                    shard_main_param.shared = model_param.shared

                # Add to group.
                model_float16_params_this_group.append(model_param)
                shard_float16_params_this_group.append(shard_model_param)
                shard_fp32_from_float16_params_this_group.append(shard_main_param)

            # fp32 params.
            elif model_param.type() == 'torch.cuda.FloatTensor':
                shard_model_param = model_param.view(-1) \
                    [param_range.start:param_range.end]
                model_fp32_params_this_group.append(model_param)
                shard_fp32_params_this_group.append(shard_model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            else:
                raise TypeError('Wrapped parameters must be one of '
                                'torch.cuda.FloatTensor,  '
                                'torch.cuda.HalfTensor, or '
                                'torch.cuda.BFloat16Tensor. '
                                'torch.npu.BFloat16Tensor. '
                                'Received {}'.format(param.type()))

        # Update optimizer's params.
        group_range["orig_group"]["params"] = [
            *shard_fp32_params_this_group,
            *shard_fp32_from_float16_params_this_group,
        ]

    return (
        model_float16_groups,
        model_fp32_groups,
        shard_float16_groups,
        shard_fp32_groups,
        shard_fp32_from_float16_groups,
    )

def DistributedDataParallelInit(
    self, module, accumulate_allreduce_grads_in_fp32,
    use_contiguous_buffers):
    args = get_args()
    super(opensora.model.distributed.DistributedDataParallel, self).__init__(module)

    self.accumulate_allreduce_grads_in_fp32 = accumulate_allreduce_grads_in_fp32
    self.use_contiguous_buffers = use_contiguous_buffers
    # If we are using fp32-accumulate-allreduce explicitly
    # this means we need main grads in a continuous buffer.
    if self.accumulate_allreduce_grads_in_fp32:
        assert self.use_contiguous_buffers

    # ===================================
    # Rest of this part applies only to
    # the case we use continuous buffers.
    # ===================================
    self._grad_buffers = None
    self._grad_buffer_param_index_map = None
    if self.use_contiguous_buffers:
        self._grad_buffers = {}
        self._grad_buffer_param_index_map = {}
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        # if args.moe:
        #     from .adaptor_parallel_state import get_expert_data_parallel_world_size
        #     expert_data_parallel_world_size = get_expert_data_parallel_world_size()

        # Simple function to define buffer type.
        def _get_buffer_type(param):
            if hasattr(param, 'moe'):
                return 'moe' + str(param.dtype)
            else:
                return torch.float \
                    if self.accumulate_allreduce_grads_in_fp32 else param.dtype

        # First calculate total number of elements per type.
        type_num_elements = {}
        for param in self.module.parameters():
            if param.requires_grad:
                dtype = _get_buffer_type(param)
                type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                           + param.data.nelement()

        # Allocate the buffer.
        for dtype, num_elements in type_num_elements.items():
            if 'moe' in str(dtype):
                num_elements_padded = expert_data_parallel_world_size * \
                    int(math.ceil(num_elements / expert_data_parallel_world_size))
                ori_dtype = eval(dtype[3:])
                self._grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                         num_elements_padded,
                                                         ori_dtype)
            else:
                num_elements_padded = data_parallel_world_size * \
                    int(math.ceil(num_elements / data_parallel_world_size))
                self._grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                         num_elements_padded,
                                                         dtype)

        # Assume the back prop order is reverse the params order,
        # store the start index for the gradients.
        for param in self.module.parameters():
            if param.requires_grad:
                dtype = _get_buffer_type(param)
                type_num_elements[dtype] -= param.data.nelement()
                param.main_grad = self._grad_buffers[dtype].get(
                    param.data.shape, type_num_elements[dtype]
                )
                if dtype not in self._grad_buffer_param_index_map:
                    self._grad_buffer_param_index_map[dtype] = {}
                self._grad_buffer_param_index_map[dtype][param] = (
                    type_num_elements[dtype],
                    type_num_elements[dtype] + param.data.nelement(),
                )

        # Backward hook.
        # Accumalation function for the gradients. We need
        # to store them so they don't go out of scope.
        self.grad_accs = []
        # Loop over all the parameters in the model.
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator functtion.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param))
                self.grad_accs.append(grad_acc)


def get_model_buffer_dp_views(self, model_buffers):
    """
    Get shard views of each of the DDP's grad buffers.

    In this nested list, the top level is grouped by the virtual model
    index and the grad buffer's data type. The sub-level is a list of
    shards of that grad buffer, where each shard in the list represents
    a contiguous view of the grad buffer, that is owned by a data-parallel
    rank. The shard boundary does not respect parameter boundaries, and
    so the elements of some parameters are split across data parallel
    ranks.

    Additionally, return references to the entire grad buffers, for use
    in _reduce_scatter_base and _all_gather_base.
    """
    args = get_args()
    data_parallel_world_size = mpu.get_data_parallel_world_size()
    # if args.moe:
    #     from .adaptor_parallel_state import get_expert_data_parallel_world_size
    #     expert_data_parallel_world_size = get_expert_data_parallel_world_size()

    # Buffer views.
    view_items = []
    for model_index, buffers in enumerate(model_buffers):
        for dtype, buf in buffers.items():
            if 'moe' in str(dtype):
                assert buf.numel() % expert_data_parallel_world_size == 0
                shard_size = int(buf.numel() / expert_data_parallel_world_size)
                buf_views = [buf[(r*shard_size):((r+1)*shard_size)]
                            for r in range(expert_data_parallel_world_size)]
                view_items.append((model_index, dtype, buf, buf_views))
            else:
                assert buf.numel() % data_parallel_world_size == 0
                shard_size = int(buf.numel() / data_parallel_world_size)
                buf_views = [buf[(r*shard_size):((r+1)*shard_size)]
                            for r in range(data_parallel_world_size)]
                view_items.append((model_index, dtype, buf, buf_views))

    return view_items


def reduce_model_grads(self, args, timers):
    """
    Reduce-scatter model grads.

    The DDP's grad buffer is used for the reduce-scatter, and thus no
    tensors are dynamically allocated.

    Note: this is a different order of reduction, versus the non-
    distributed optimizer, which reduces: 1) layernorm grads, 2) all
    grads, 3) embedding grads.
    """

    # All-reduce layer-norm grads (for sequence parallelism).
    timers('layernorm-grads-all-reduce', log_level=1).start(
         barrier=args.barrier_with_L1_time
    )
    self.allreduce_layernorm_grads(args)
    timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads.
    timers('embedding-grads-all-reduce', log_level=1).start(
         barrier=args.barrier_with_L1_time
    )
    self.allreduce_embedding_grads(args)
    timers('embedding-grads-all-reduce').stop()

    # Reduce-scatter setup.
    timers('grads-reduce-scatter', log_level=1).start(
         barrier=args.barrier_with_L1_time
    )
    data_parallel_rank = mpu.get_data_parallel_rank()
    data_parallel_world_size = mpu.get_data_parallel_world_size()
    data_parallel_group = mpu.get_data_parallel_group()
    # if args.moe:
    #     from .adaptor_parallel_state import get_expert_data_parallel_world_size, get_expert_data_parallel_rank, \
    #         get_expert_data_parallel_group
    #     expert_data_parallel_group = get_expert_data_parallel_group()
    #     expert_data_parallel_rank = get_expert_data_parallel_rank()
    #     expert_data_parallel_world_size = get_expert_data_parallel_world_size()

    # Scale grad buffers by '1 / data_parallel_world_size'.
    for model in self.models:
        for dtype, gbuf in model._grad_buffers.items():
            if 'moe' in str(dtype):
                gbuf.data /= expert_data_parallel_world_size
            else:
                gbuf.data /= data_parallel_world_size

    # Reduce-scatter all grads.
    gbuf_view_items = self.get_model_grad_buffer_dp_views()
    for index, (model_index, dtype, gbuf, gbuf_views) \
        in enumerate(gbuf_view_items):
        if 'moe' in str(dtype):
            torch.distributed._reduce_scatter_base(
                gbuf_views[expert_data_parallel_rank],
                gbuf,
                group=expert_data_parallel_group
            )
        else:
            torch.distributed._reduce_scatter_base(
                gbuf_views[data_parallel_rank],
                gbuf,
                group=data_parallel_group
            )

    timers('grads-reduce-scatter').stop()



def gather_model_params(self, args, timers):
    """
    All-gather updated model params.

    The DDP's grad buffer is used for the all-gather, and thus no
    tensors are dynamically allocated. After the all-gather, the params
    can be copied from param.main_grad to param.
    """

    timers('params-all-gather', log_level=1).start(
         barrier=args.barrier_with_L1_time
    )

    data_parallel_rank = mpu.get_data_parallel_rank()
    data_parallel_group = mpu.get_data_parallel_group()
    # if args.moe:
    #     from .adaptor_parallel_state import get_expert_data_parallel_world_size, get_expert_data_parallel_rank, \
    #         get_expert_data_parallel_group
    #     expert_data_parallel_group = get_expert_data_parallel_group()
    #     expert_data_parallel_rank = get_expert_data_parallel_rank()
    #     expert_data_parallel_world_size = get_expert_data_parallel_world_size()

    # All-gather updated main params.
    # - All grad buffer views are guaranteed to have the same num elements
    #   across all data parallel ranks, with grad buffer padding that is done
    #   in distributed.py. Thus, all sub-views will have consistent start/end
    #   indexes across data parallel ranks.
    pbuf_view_items = self.get_model_param_buffer_dp_views()
    for index, (model_index, dtype, pbuf, pbuf_views) \
        in enumerate(pbuf_view_items):
        if 'moe' in str(dtype):
            torch.distributed._all_gather_base(
                pbuf,
                pbuf_views[expert_data_parallel_rank],
                group=expert_data_parallel_group
            )
        else:
            torch.distributed._all_gather_base(
                pbuf,
                pbuf_views[data_parallel_rank],
                group=data_parallel_group
            )

    # Copy from param buffer to each param.
    for model_id, model in enumerate(self.models):
        for dtype, param_map in model._grad_buffer_param_index_map.items():
            for param, buf_range in param_map.items():
                param_buf = self.param_buffers[model_id][dtype]
                param_buf_shard = param_buf[buf_range[0]:buf_range[1]]
                param.view(-1).detach().copy_(param_buf_shard)

    timers('params-all-gather').stop()


opensora.model.distributed.DistributedDataParallel.__init__ = DistributedDataParallelInit
opensora.optimizer.distrib_optimizer.DistributedOptimizer.__init__ = DistributedOptimizerInit
opensora.optimizer.distrib_optimizer.DistributedOptimizer.build_model_and_main_param_groups = build_model_and_main_param_groups
opensora.optimizer.distrib_optimizer.DistributedOptimizer.get_model_buffer_dp_views = get_model_buffer_dp_views
opensora.optimizer.distrib_optimizer.DistributedOptimizer.reduce_model_grads = reduce_model_grads
opensora.optimizer.distrib_optimizer.DistributedOptimizer.gather_model_params = gather_model_params