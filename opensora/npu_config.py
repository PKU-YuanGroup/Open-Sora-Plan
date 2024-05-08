import math
import mmap
import os
import pickle
import random
import numpy as np
import torch

try:
    import torch_npu
    npu_is_available = True
    from torch_npu.contrib import transfer_to_npu
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.runtime.utils import get_global_norm
    from deepspeed import comm as dist
except:
    npu_is_available = False
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import is_moe_param
from deepspeed.runtime.utils import (bwc_tensor_model_parallel_rank, get_global_norm, empty_cache, see_memory_usage,
                                     inf, is_model_parallel_parameter, align_dense_tensors, all_gather_dp_groups)
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.utils import groups
from deepspeed.utils import logger
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler

rank = int(os.environ["RANK"])
OPTIMIZER_ALLGATHER_TIMER = 'optimizer_allgather'
OPTIMIZER_GRADIENTS_TIMER = 'optimizer_gradients'
OPTIMIZER_STEP_TIMER = 'optimizer_step'
OPTIMIZER_TIMERS = [OPTIMIZER_ALLGATHER_TIMER, OPTIMIZER_GRADIENTS_TIMER, OPTIMIZER_STEP_TIMER]


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()

def print_with_rank(msg):
    print(f"[RANK-{rank}]: {msg}", flush=True)

def __init__(self,
             init_optimizer,
             param_names,
             timers,
             static_loss_scale=1.0,
             dynamic_loss_scale=False,
             dynamic_loss_args=None,
             verbose=True,
             contiguous_gradients=True,
             reduce_bucket_size=500000000,
             use_multi_rank_bucket_allreduce=True,
             allgather_bucket_size=5000000000,
             dp_process_group=None,
             expert_parallel_group=None,
             expert_data_parallel_group=None,
             reduce_scatter=True,
             overlap_comm=False,
             offload_optimizer_config=None,
             mpu=None,
             clip_grad=0.0,
             gradient_accumulation_dtype=torch.float32,
             communication_data_type=torch.float16,
             postscale_gradients=True,
             gradient_predivide_factor=1.0,
             gradient_accumulation_steps=1,
             ignore_unused_parameters=True,
             partition_grads=True,
             round_robin_gradients=False,
             has_moe_layers=False,
             fp16_master_weights_and_gradients=False,
             elastic_checkpoint=False):
    if offload_optimizer_config is not None and offload_optimizer_config.device != OffloadDeviceEnum.none:
        self.cpu_offload = True
        self.cpu_offload_pin_memory = offload_optimizer_config.pin_memory
    else:
        self.cpu_offload = False
        self.cpu_offload_pin_memory = False

    if dist.get_rank() == 0:
        logger.info(f"Reduce bucket size {reduce_bucket_size}")
        logger.info(f"Allgather bucket size {allgather_bucket_size}")
        logger.info(f"CPU Offload: {self.cpu_offload}")
        logger.info(f'Round robin gradient partitioning: {round_robin_gradients}')
    # The fused optimizer does all the work. We need this layer for two reason:
    # 1. maintain same user API from apex.fp16_utils
    # 2. keep common stuff here in case we need to add ne552w fused optimizer later

    self.elastic_checkpoint = elastic_checkpoint
    self.param_names = param_names
    self.mpu = mpu
    # differences from apex.fp16_utils:
    # - assume all model params in fp16
    # - assume all params requires grad
    # - flat by groups, not keeping state. TODO: remove state explicitly?
    # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
    if not get_accelerator().is_available():
        raise SystemError("Accelerator is not detected, cannot perform low precision training (e.g., fp16, bf16).")
    self.optimizer = init_optimizer

    # Use torch (un)flatten ops
    self.flatten = _flatten_dense_tensors
    self.unflatten = _unflatten_dense_tensors

    # ZeRO stage 1 (False) or 2 (True)
    self.partition_gradients = partition_grads
    self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"

    self.timers = timers

    self.reduce_scatter = reduce_scatter

    self.overlap_comm = overlap_comm

    self.deepspeed_adam_offload = self.cpu_offload

    self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'

    self.dp_process_group = dp_process_group
    self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
    # expert parallel group
    self.ep_process_group = expert_parallel_group

    # data parallel group for experts
    self.expert_dp_process_group = expert_data_parallel_group

    # data parallel size for non-experts
    dp_size = dist.get_world_size(group=self.dp_process_group)

    # For MoE models this maybe different for different param group
    # It will be modified during MoE setup later in the init
    self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
    self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]

    self.is_gradient_accumulation_boundary = True

    # CPU-Offload requires contiguous gradients
    self.contiguous_gradients = contiguous_gradients or self.cpu_offload

    self.has_moe_layers = has_moe_layers
    if self.has_moe_layers:
        self._configure_moe_settings()
    self._global_grad_norm = 0.

    if mpu is None:
        self.model_parallel_group = None
        self.model_parallel_world_size = 1
        self.model_parallel_rank = 0
    else:
        self.model_parallel_group = mpu.get_model_parallel_group()
        self.model_parallel_world_size = mpu.get_model_parallel_world_size()
        self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

    self.overflow = False
    self.clip_grad = clip_grad
    self.communication_data_type = communication_data_type
    self.gradient_predivide_factor = gradient_predivide_factor
    self.postscale_gradients = postscale_gradients
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.micro_step_id = 0
    self.ignore_unused_parameters = ignore_unused_parameters
    self.round_robin_gradients = round_robin_gradients

    self.extra_large_param_to_reduce = None
    self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients

    if self.fp16_master_weights_and_gradients:
        assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], \
            f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32." \
            f"Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}." \
            f"Either disable fp16_master_weights_and_gradients or enable {self.zero_stage_string} Offload with DeepSpeedCPUAdam."

    if self.reduce_scatter:
        valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        assert self.communication_data_type in valid_reduce_scatter_dtypes, f"{self.zero_stage_string} supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
        assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
        assert self.postscale_gradients, "pre-scale gradients is not yet supported with {self.zero_stage_string} with reduce scatter enabled"

    # param flattened by groups
    self.bit16_groups = []
    self.bit16_groups_flat = []

    # param partitioned by data parallel degree
    # this will contain a list of equal sized tensors
    # each of which will be updated by a different process
    self.parallel_partitioned_bit16_groups = []

    # a single 32-bit partition of the parallel partitioned parameters
    # that this process will update
    self.single_partition_of_fp32_groups = []

    # param partition info

    # These are the parameters in each group that will not be updated by this process directly
    self.params_not_in_partition = []

    # These are the parameters that will be updated by this process directly
    self.params_in_partition = []

    # Offset from the first parameter in the self.params_in_partition
    # the parameter boundaries may not align with partition boundaries
    # so we need to keep track of the offset
    self.first_offset = []

    # number of elements per partition in each group
    self.partition_size = []

    # align nccl all-gather send buffers to 4-byte boundary
    self.nccl_start_alignment_factor = 512  # 4-byte alignment/sizeof(fp16) = 2
    print_with_rank(f"set self.nccl_start_alignment_factor {self.nccl_start_alignment_factor}")

    assert (
            allgather_bucket_size % self.nccl_start_alignment_factor == 0
    ), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

    self.all_reduce_print = False
    self.dtype = self.optimizer.param_groups[0]['params'][0].dtype
    self.gradient_accumulation_dtype = gradient_accumulation_dtype

    if self.dtype != self.gradient_accumulation_dtype:
        self.use_separate_grad_accum = True
    else:
        self.use_separate_grad_accum = False
    if self.use_separate_grad_accum and not self.partition_gradients:
        self.use_grad_accum_attribute = True
    else:
        self.use_grad_accum_attribute = False

    self.round_robin_bit16_groups = []
    self.round_robin_bit16_indices = []

    # Use different parallel to do all_to_all_reduce related things
    # padding on each partition for alignment purposes
    self.groups_padding = []
    # loop to deal with groups
    for i, param_group in enumerate(self.optimizer.param_groups):
        partition_id = dist.get_rank(group=self.real_dp_process_group[i])

        # push this group to list before modify
        # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
        trainable_parameters = []
        for param in param_group['params']:
            if param.requires_grad:
                param.grad_accum = None
                trainable_parameters.append(param)
        self.bit16_groups.append(trainable_parameters)

        # not sure why apex was cloning the weights before flattening
        # removing cloning here

        see_memory_usage(f"Before moving param group {i} to CPU")
        # move all the parameters to cpu to free up GPU space for creating flat buffer
        move_to_cpu(self.bit16_groups[i])
        empty_cache()
        see_memory_usage(f"After moving param group {i} to CPU", force=False)

        # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
        # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
        # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
        # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
        if self.round_robin_gradients:
            round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
        else:
            round_robin_tensors = self.bit16_groups[i]
            round_robin_indices = list(range(len(self.bit16_groups[i])))

        self.round_robin_bit16_groups.append(round_robin_tensors)
        self.round_robin_bit16_indices.append(round_robin_indices)

        # create flat buffer in CPU and move to GPU
        self.bit16_groups_flat.append(
            self.flatten_dense_tensors_aligned(
                self.round_robin_bit16_groups[i],
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i])).to(
                get_accelerator().current_device_name()))
        see_memory_usage(f"After flattening and moving param group {i} to GPU", force=False)

        # Record padding required for alignment
        if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
            padding = self.bit16_groups_flat[i].numel() - sum(
                [t.numel() for t in self.round_robin_bit16_groups[i]])
        else:
            padding = 0
        self.groups_padding.append(padding)

        if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
            see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=False)

        # set model bit16 weight to slices of flattened buffer
        self._update_model_bit16_weights(i)

        # divide the flat weights into near equal partition equal to the data parallel degree
        # each process will compute on a different part of the partition
        data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
        self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

        # verify that data partition start locations are 4-byte aligned
        for partitioned_data in data_parallel_partitions:
            assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

        # A partition of the fp32 master weights that will be updated by this process.
        # Note that the params in single_partition_of_fp32_groups is cloned and detached
        # from the origin params of the model.
        if not fp16_master_weights_and_gradients:
            self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(
                self.device).clone().float().detach())
        else:
            self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(
                self.device).clone().half().detach())

        # Set local optimizer to have flat params of its own partition.
        # After this, the local optimizer will only contain its own partition of params.
        # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
        self.single_partition_of_fp32_groups[
            i].requires_grad = True  # keep this in case internal optimizer uses it
        param_group['params'] = [self.single_partition_of_fp32_groups[i]]

        partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
        params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
            self.round_robin_bit16_groups[i], partition_size, partition_id)

        self.partition_size.append(partition_size)
        self.params_in_partition.append(params_in_partition)
        self.params_not_in_partition.append(params_not_in_partition)
        self.first_offset.append(first_offset)

    self.reduce_bucket_size = int(reduce_bucket_size)
    self.use_multi_rank_bucket_allreduce = use_multi_rank_bucket_allreduce
    self.allgather_bucket_size = int(allgather_bucket_size)

    self.reduction_stream = None if get_accelerator().is_synchronized_device() else get_accelerator().Stream()
    # self.copy_grad_stream = get_accelerator().Stream()
    self.callback_queued = False

    self.param_dict = {}

    # map between param_id and bool to specify if a param is in this partition
    self.is_param_in_current_partition = {}

    self.grads_in_ipg_bucket = []
    self.params_in_ipg_bucket = []
    self.elements_in_ipg_bucket = 0
    self.params_already_reduced = []
    self._release_ipg_buffers()
    self.previous_reduced_grads = None
    self.ipg_bucket_has_moe_params = False

    # simplified param id
    self.param_id = {}

    # interesting code: unique ids being assigned to individual parameters
    largest_param_numel = 0
    count = 0
    for i, params_group in enumerate(self.bit16_groups):
        for param in params_group:
            unique_id = id(param)
            self.param_id[unique_id] = count
            self.param_dict[count] = param
            self.params_already_reduced.append(False)
            if param.numel() > largest_param_numel:
                largest_param_numel = param.numel()
            count = count + 1

    for param_group in self.params_in_partition:
        for param in param_group:
            self.is_param_in_current_partition[self.get_param_id(param)] = True

    for param_group in self.params_not_in_partition:
        for param in param_group:
            self.is_param_in_current_partition[self.get_param_id(param)] = False

    if self.cpu_offload:
        self.accumulated_grads_in_cpu = {}
        self.norm_for_param_grads = {}
        self.local_overflow = False
        self.grad_position = {}
        self.temp_grad_buffer_for_cpu_offload = torch.zeros(largest_param_numel,
                                                            device=self.device,
                                                            dtype=self.dtype)
        if self.cpu_offload_pin_memory:
            self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(
                self.temp_grad_buffer_for_cpu_offload)
        self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                            device=get_accelerator().current_device_name(),
                                                            dtype=self.dtype)
        for i, params_group in enumerate(self.bit16_groups):
            self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

    # mapping from parameter to partition that it belongs to
    self.param_to_partition_ids = {}

    # stores if a partition has been reduced in this step
    self.is_partition_reduced = {}

    # number of grads in partition that still need to be computed
    self.remaining_grads_in_partition = {}

    # total number of grads in partition
    self.total_grads_in_partition = {}

    # stores if a grad in a partition has been computed or not
    self.is_grad_computed = {}

    # stores the offset at which a parameter gradient needs to be inserted in a partition
    self.grad_partition_insertion_offset = {}

    # the offset in the gradient at which it must be inserted at the beginning of the partition
    self.grad_start_offset = {}

    # will store the averaged gradients required by this partition
    self.averaged_gradients = {}

    # For cpu_offload, will store the averaged gradients required by this partition
    self.offload_gradient_dict = {}

    # store index of first parameter in each partition
    self.first_param_index_in_partition = {}

    # initializes all data structures for implementing gradient partitioning
    self.initialize_gradient_partitioning_data_structures()

    # resets the data structure value for the next backward propagation
    self.reset_partition_gradient_structures()

    # creates backward hooks for gradient partitioning
    if self.partition_gradients or self.overlap_comm:
        self.create_reduce_and_remove_grad_hooks()

    self.custom_loss_scaler = False
    self.external_loss_scale = None

    # we may have a way of fusing dynamic scale. Do not support for now
    self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                        static_loss_scale=static_loss_scale,
                                        dynamic_scaling=dynamic_loss_scale,
                                        dynamic_loss_args=dynamic_loss_args)
    self.dynamic_loss_scale = self.loss_scaler.dynamic

    if self.dtype != torch.float16:
        # Only fp16 should use dynamic loss scaling
        assert self.loss_scaler.cur_scale == 1.0
        assert not self.dynamic_loss_scale

    see_memory_usage("Before initializing optimizer states", force=True)
    self.initialize_optimizer_states()
    see_memory_usage("After initializing optimizer states", force=True)

    if dist.get_rank() == 0:
        logger.info(f"optimizer state initialized")

    if dist.get_rank(group=self.dp_process_group) == 0:
        see_memory_usage(f"After initializing ZeRO optimizer", force=True)

    self._link_all_hp_params()
    self._enable_universal_checkpoint()
    self._param_slice_mappings = self._create_param_mapping()

def step(self, closure=None):
    """
    Not supporting closure.
    """
    self.micro_step_id = -1

    see_memory_usage(f"In step before checking overflow")

    # First compute norm for all group so we know if there is overflow
    if self.dtype == torch.float16:
        self.check_overflow()

    prev_scale = self.loss_scale
    self._update_scale(self.overflow)
    if self.overflow:
        see_memory_usage('After overflow before clearing gradients')
        self.zero_grad(set_to_none=True)
        if self.cpu_offload:
            self.reset_cpu_buffers()
        else:
            self.averaged_gradients = {}

        see_memory_usage('After overflow after clearing gradients')

        for timer in OPTIMIZER_TIMERS:
            self.timers(timer).start()
            self.timers(timer).stop()
        return

    # Step 1:- Calculate gradient norm using bit-16 grads
    see_memory_usage('Before norm calculation')
    scaled_global_grad_norm = self.scaled_global_norm()
    self._global_grad_norm = scaled_global_grad_norm / prev_scale
    see_memory_usage('After norm before optimizer')

    # Step 2:- run optimizer and upscaling simultaneously
    for i, group in enumerate(self.bit16_groups):
        self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
        partition_id = dist.get_rank(group=self.real_dp_process_group[i])
        if self.cpu_offload:
            single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

            self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()
            self.timers(OPTIMIZER_STEP_TIMER).start()
            self._optimizer_step(i)

            # Disabled, this is not currently working
            # from deepspeed.ops.adam import DeepSpeedCPUAdam
            # if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
            #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            #    fp32_partition = self.single_partition_of_fp32_groups[i]
            #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.single_partition_of_fp32_groups[i]
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)

            self.timers(OPTIMIZER_STEP_TIMER).stop()
        else:
            # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
            self.free_grad_in_param_list(self.params_not_in_partition[i])

            # create a flat gradients for parameters updated by this process
            # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                single_grad_partition = self.flatten_dense_tensors_aligned(
                    self.averaged_gradients[i],
                    int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
            else:
                single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                    self.single_partition_of_fp32_groups[i].dtype)
            assert single_grad_partition.numel() == self.partition_size[i], \
                "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                    single_grad_partition.numel(), self.partition_size[i], i, partition_id)

            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
            # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
            self.free_grad_in_param_list(self.params_in_partition[i])

            self.averaged_gradients[i] = None

            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

            self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

            # Step 3:- run the optimizer if no offloading
            self.timers(OPTIMIZER_STEP_TIMER).start()
            self._optimizer_step(i)
            # Step 4:- get rid of the fp32 gradients. Not needed anymore
            self.single_partition_of_fp32_groups[i].grad = None
            del single_grad_partition
            bit16_partitions = self.parallel_partitioned_bit16_groups[i]
            fp32_partition = self.single_partition_of_fp32_groups[i]
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)
            self.timers(OPTIMIZER_STEP_TIMER).stop()

    see_memory_usage('After optimizer before all-gather')
    if self.cpu_offload:
        self.reset_cpu_buffers()

    self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
    # Gather the updated weights from everyone.
    # Then all partitions of the model parameters are updated and ready for next round forward.
    all_gather_dp_groups(groups_flat=self.bit16_groups_flat,
                         partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                         dp_process_group=self.real_dp_process_group,
                         start_alignment_factor=self.nccl_start_alignment_factor,
                         allgather_bucket_size=self.allgather_bucket_size)
    # print_with_rank("step fcn: Close all gather")
    self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

    # TODO: we probably don't need this? just to be safe
    for i in range(len(self.bit16_groups)):
        self._update_model_bit16_weights(i)

    self.timers.log(OPTIMIZER_TIMERS)
    see_memory_usage('After zero_optimizer step')

    return

# def scaled_global_norm(self, norm_type=2):
#     assert norm_type == 2, "only L2 norm supported"
#     norm_groups = []
#     for i, group in enumerate(self.bit16_groups):
#         partition_id = dist.get_rank(group=self.real_dp_process_group[i])
#         if self.cpu_offload:
#             norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]))
#             single_grad_partition = self.single_partition_of_fp32_groups[i].grad
#         else:
#             norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))
#
#     if self.has_moe_layers:
#         self._average_expert_grad_norms(norm_groups)
#
#     # note that the get_global_norm function only supports l2 norm
#     return get_global_norm(norm_list=norm_groups)
def allreduce_bucket(self, bucket, rank=None, log=None, divide=True, process_group=None):
    rank = None
    tensor = self.flatten(bucket)

    process_group = self.dp_process_group if process_group is None else process_group

    tensor_to_allreduce = tensor


    if False or self.sequence_parallel_size > 1:
        communication_data_type = torch.float32
    else:
        communication_data_type = self.communication_data_type
    # print_with_rank(f"allreduce_bucket fcn: communication_data_type is {communication_data_type}")

    if communication_data_type != tensor.dtype:
        tensor_to_allreduce = tensor.to(communication_data_type)

    # print_with_rank(f"allreduce_bucket fcn: tensor_to_allreduce.dtype is {tensor_to_allreduce.dtype}")

    if divide:
        # print_with_rank(f"allreduce_bucket RUN DIV")
        tensor_to_allreduce.div_(dist.get_world_size(group=process_group) / float(self.sequence_parallel_size))

    if rank is None:
        #    "All Reducing"
        dist.all_reduce(tensor_to_allreduce, group=process_group)
    else:
        global_rank = dist.get_global_rank(process_group, rank)
        dist.reduce(tensor_to_allreduce, global_rank, group=process_group)

    if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
        if rank is None or rank == dist.get_rank(group=process_group):
            tensor.copy_(tensor_to_allreduce)

    return tensor

def average_tensor(self, tensor):
    if self.overlap_comm:
        stream = self.reduction_stream
        if not get_accelerator().is_synchronized_device():
            stream.wait_stream(get_accelerator().current_stream())
    else:
        stream = get_accelerator().current_stream()

    with get_accelerator().stream(stream):
        if not self.reduce_scatter:
            self.gradient_reduction_w_predivide(tensor)
            return

        # Accumulate destination ranks and bucket offsets for each gradient slice.
        # Note: potential future optimization, record access pattern of parameters
        # in backward pass and partition gradients w.r.t. access pattern so that our
        # bucket is guaranteed to be contiguous w.r.t. ranks
        rank_and_offsets = []
        real_dp_process_group = []
        curr_size = 0
        prev_id, prev_process_group = -1, None

        process_group = self.dp_process_group
        # count = 0
        for i, param, param_id in self.params_in_ipg_bucket:

            process_group = self.dp_process_group
            grad_reduc = self.get_gradient_for_reduction(param)
            # Averages gradients at parameter level if ipg has a moe param
            # Otherwise averaging is done at the entire buffer level at the end of the loop
            # MoE param have different groups
            if self.ipg_bucket_has_moe_params:
                process_group = self.expert_dp_process_group[param.group_name] if is_moe_param(
                    param) else self.dp_process_group
                grad_reduc.data.div_(dist.get_world_size(group=process_group) / float(self.sequence_parallel_size))

            partition_ids = self.param_to_partition_ids[i][param_id]
            assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids
                        ]), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
            partition_size = self.partition_size[i]
            # Get all partition ids + their offsets
            partition_ids_w_offsets = []
            for partition_id in partition_ids:
                offset = self.grad_start_offset[i][partition_id][param_id]
                partition_ids_w_offsets.append((partition_id, offset))
            partition_ids_w_offsets.sort(key=lambda t: t[1])

            # Calculate rank and offsets for grad slices
            for idx in range(len(partition_ids_w_offsets)):
                partition_id, offset = partition_ids_w_offsets[idx]

                # if dist.get_rank() == 0 and count < 100:
                #     print(f"Rank {dist.get_rank()} rank offset id {idx} calculated dp size {dist.get_world_size(group=process_group)} real dp size {dist.get_world_size(self.real_dp_process_group[i])} and dst: {partition_id}")
                # count += 1

                # Calculate numel for grad slice depending on partition location
                if idx == len(partition_ids_w_offsets) - 1:
                    # Last partition_id uses its own offset
                    numel = param.numel() - offset
                else:
                    # Set numel to next partition's offset
                    numel = partition_ids_w_offsets[idx + 1][1] - offset

                # Merge bucket ranges if they belong to the same rank
                if partition_id == prev_id and process_group == prev_process_group:
                    prev_pid, prev_size, prev_numel = rank_and_offsets[-1]
                    rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                else:
                    rank_and_offsets.append((partition_id, curr_size, numel))
                    real_dp_process_group.append(process_group)
                curr_size += numel
                prev_id, prev_process_group = partition_id, process_group

        if not self.ipg_bucket_has_moe_params:
            tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))

        buckets = {}
        for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
            grad_slice = tensor.narrow(0, int(bucket_offset), int(numel))
            bucket_key = real_dp_process_group[i] if self.use_multi_rank_bucket_allreduce else (
                dst, real_dp_process_group[i])
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            if self.use_multi_rank_bucket_allreduce:
                buckets[bucket_key].append((dst, grad_slice))
            else:
                buckets[bucket_key].append(grad_slice)

        for bucket_key in buckets:
            if self.use_multi_rank_bucket_allreduce:
                self.allreduce_and_scatter(buckets[bucket_key],
                                           numel_per_bucket=self.reduce_bucket_size,
                                           divide=self.ipg_bucket_has_moe_params,
                                           process_group=bucket_key)
            else:
                dst, process_group = bucket_key
                self.allreduce_no_retain(buckets[bucket_key],
                                         numel_per_bucket=self.reduce_bucket_size,
                                         rank=dst,
                                         divide=self.ipg_bucket_has_moe_params,
                                         process_group=process_group)

def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
    # compute combined scale factor for this group
    combined_scale = self.loss_scale
    # print(f"Enter line 42. loss_scale  is {self.loss_scale}, total_norm is {total_norm}", flush=True)
    if self.clip_grad > 0.:
        # print("Enter line 44. self.clip_grad > 0 is true", flush=True)
        # norm is in fact norm*scale
        clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
        if clip > 1:
            combined_scale = clip * self.loss_scale
            # print("Enter line 49. clip > 1 is true", flush=True)

    # print("Enter line 53, combined scale is {}".format(combined_scale), flush=True)
    for grad in grad_groups_flat:
        if isinstance(grad, list):
            sub_partitions = grad
            for g in sub_partitions:
                g.data.mul_(1. / combined_scale)
        else:
            grad.data.mul_(1. / combined_scale)




class NPUConfig:
    N_NPU_PER_NODE = 8

    def __init__(self):
        self.on_npu = npu_is_available
        self.node_world_size = self.N_NPU_PER_NODE
        self.profiling = False
        self.profiling_step = 5
        self.enable_FA = True
        self.enable_FP32 = False
        self.load_pickle = True
        self.use_small_dataset = False
        if self.use_small_dataset:
            self.load_pickle = False

        self._loss = []
        self.work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pickle_save_path = f"{self.work_path}/pickles"
        self.mm = dict()

        if self.on_npu:
            torch_npu.npu.set_compile_mode(jit_compile=False)
            DeepSpeedZeroOptimizer.unscale_and_clip_grads = unscale_and_clip_grads
            DeepSpeedZeroOptimizer.average_tensor = average_tensor
            DeepSpeedZeroOptimizer.allreduce_bucket = allreduce_bucket
            DeepSpeedZeroOptimizer.step = step
            DeepSpeedZeroOptimizer.__init__ = __init__

        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.rank = torch.cuda.current_device()
            self.world_size = self.N_NPU_PER_NODE
        self.print_with_rank(f"The npu_config.on_npu is {self.on_npu}")

    def get_node_size(self):
        return self.world_size / self.node_world_size

    def get_local_rank(self):
        return self.rank % self.N_NPU_PER_NODE

    def get_pickle_path(self, file_name):
        return f"{self.pickle_save_path}/{file_name}_local.pkl"

    def free_mm(self):
        for key, value in self.mm.items():
            value.close()
        self.mm.clear()

    def __del__(self):
        self.free_mm()

    def try_load_pickle(self, file_name, function):
        file_name = self.get_pickle_path(file_name)
        if os.path.exists(file_name) and self.load_pickle:
            with open(file_name, 'rb') as file:
                # self.mm[file_name] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                # # 使用 mmap 进行数据读取
                # loaded_data = pickle.loads(self.mm[file_name][:])
                loaded_data = pickle.load(file)
                return loaded_data
        else:
            data = function()
            if not self.use_small_dataset:
                if self.rank % self.N_NPU_PER_NODE == 0:
                    # 只需要rank0保存文件
                    os.makedirs(self.pickle_save_path, exist_ok=True)
                    with open(file_name, 'wb') as file:
                        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            return data

    def npu_format_cast(self, x):
        return torch_npu.npu_format_cast(x, 2)

    def calc_grad_norm(self, model):
        # 计算并打印梯度范数
        # model_engine = accelerator.deepspeed_engine_wrapped.engine
        # gradients = model_engine.get_gradients()
        # grad_norm = get_grad_norm(gradients)
        # 计算并打印梯度范数
        grad_norm = 0
        n_grad = 0
        for name, param in model.named_parameters():
            grad_data = deepspeed.utils.safe_get_full_grad(param)
            # self.print_tensor_stats(grad_data, name=name)

            if grad_data is not None:
                param_norm = grad_data.norm(2)
                grad_norm += param_norm.item() ** 2
                n_grad += 1
        grad_norm = (grad_norm / n_grad) ** (1. / 2)

        return grad_norm

    def _run(self, operator, x, tmp_dtype, out_dtype=None, out_nd_format=False):
        if self.on_npu:
            if out_dtype is None:
                out_dtype = x.dtype

            with torch.cuda.amp.autocast(enabled=False):
                x = operator.to(tmp_dtype)(x.to(tmp_dtype))
                x = x.to(out_dtype)
                if out_nd_format:
                    return self.npu_format_cast(x)
                else:
                    return x
        else:
            return operator(x)

    def run_group_norm(self, operator, x):
        return self._run(operator, x, torch.float32)

    def print_tensor_stats(self, tensor, name="Tensor"):
        if tensor is None:
            self.print_msg(f"Tensor {name} is None.")
            return

        x_dtype = tensor.dtype
        tensor = tensor.to(torch.bfloat16)
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        abs_max_val = min(abs(max_val), abs(min_val))
        mean_val = tensor.mean().item()
        median_val = tensor.median().item()
        std_val = tensor.std().item()
        shape = tensor.shape
        self.print_msg(
            f"{name} - Max: {max_val}, Min: {min_val}, Mean: {mean_val}, AbsMax: {abs_max_val},"
            f"Median: {median_val}, Std: {std_val}, Shape: {shape}, Type: {x_dtype}")

    def run_conv3d(self, operator, x, out_dtype):
        return self._run(operator, x, torch.float16, out_dtype, out_nd_format=True)

    def run_pool_2d(self, operator, x):
        return self._run(operator, x, torch.float16)

    def seed_everything(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def print_with_rank(self, msg, rank=0, save=False):
        if self.rank == rank:
            print(f"{msg}", flush=True)
            if save:
                self._loss.append(msg)

    def print_msg(self, msg, on=True, rank=None):
        if on:
            if self.rank == rank or rank is None:
                print(f"[RANK-{self.rank}]: {msg}", flush=True)

    def save_loss(self, filename, rank=0):
        if self.rank == rank:
            import json
            with open(filename, 'w') as file:
                json.dump(self._loss, file, indent=4)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                     scale=None) -> torch.Tensor:
        # L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_bias = torch.zeros_like(attn_weight, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.zeros_like(attn_weight, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-10000.0"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-10000.0"))
            else:
                attn_bias += attn_mask

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    def print_tensor_with_rank(self, name, tensor, rank=0, dim_print_cnt=[]):
        if self.rank == rank:
            print(name)
            print(tensor.size())
            def print_dim(tensor_, indices):
                if tensor_.dim() == len(indices):
                    print('{0:10.5f}'.format(tensor[tuple(indices)].detach().item()), end=' ')
                else:
                    cur_dim = len(indices)
                    for x in range(0, tensor_.size(cur_dim), tensor_.size(cur_dim) // dim_print_cnt[cur_dim]):
                        print_dim(tensor_, indices + [x])
                    print()
            print_dim(tensor, [])


npu_config = NPUConfig()
