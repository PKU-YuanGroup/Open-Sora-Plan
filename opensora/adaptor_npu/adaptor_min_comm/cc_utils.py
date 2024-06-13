import torch
from enum import Enum
from .min_comm_cfg import min_comm_config, ModuleType


def check_equal(a, b, error_info):
    if a != b:
        if torch.npu.current_device() == 0:
            print(error_info)

def print_tensor_value(name, value, device_id=0):
    if min_comm_config.print_tensor_value_enabled and torch.cuda.current_device() == device_id:
        n = min_comm_config.parallel_num * min_comm_config.tp_world_size
        per = value.shape[0] // n
        slices = []
        for k in range(n):
            v = torch.flatten(value[k * per: (k + 1) * per])
            slices.append(v[:5])
        print(f"{name}, shape={value.shape}, value=\n{torch.cat(tuple(slices)).view(n, -1)}", flush=True)


def set_context(ctx, input_, weight, bias):
    ctx.save_for_backward(input_, weight)
    ctx.use_bias = bias is not None


def infer_matmul_out_shape(shape_a, shape_b):
    shape_a[-1] = shape_b[-1]
    return shape_a


def reshape_to_2D(input_tensor):
    # Convert the tensor shapes to 2D for execution compatibility
    input_tensor = input_tensor.reshape(input_tensor.shape[0] * input_tensor.shape[1],
                                        input_tensor.shape[2])
    return input_tensor


def async_gather_along_first_dim(input_, group, world_size):
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output_ = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device(), requires_grad=False)
    work = torch.distributed._all_gather_base(output_, input_.contiguous(), group=group, async_op=True)
    return work, output_


def shuffle_as_cc_reduce_scatter(input_, world_size, parallel_num):
    per = input_.shape[0] // parallel_num // world_size
    input_shape = list(input_.shape)
    reshape_tensor = torch.reshape(input_, [parallel_num, world_size, per] + input_shape[1:])
    return torch.reshape(reshape_tensor.transpose(0, 1), tuple(input_shape))


def shuffle_as_cc_all_gather(input_, world_size, parallel_num):
    per = input_.shape[0] // parallel_num // world_size
    input_shape = list(input_.shape)
    reshape_tensor = torch.reshape(input_, [world_size, parallel_num, per] + input_shape[1:])
    return torch.reshape(reshape_tensor.transpose(0, 1), tuple(input_shape))


def is_grad_needed(needs_input_grad):
    is_grad_input_needed, is_grad_weight_needed, is_grad_bias_needed = needs_input_grad
    if not is_grad_input_needed:
        raise RuntimeError("To use CC, grad_input is necessary to compute.")
    if not is_grad_weight_needed and is_grad_bias_needed:
        raise RuntimeError("To use CC, grad_weight must be needed if grad_bias is required.")
    return is_grad_weight_needed, is_grad_bias_needed


def get_parallel_num(m, k, n):
    parallel_num = min_comm_config.parallel_num
    shape_str = str([m, k, n])
    if len(min_comm_config.customized_cc_dict) > 0 and str(shape_str) in min_comm_config.customized_cc_dict.keys():
        parallel_num = min_comm_config.customized_cc_dict.get(shape_str)
    return parallel_num


class CommunicationType(Enum):
    ALL_GATHER = 0
    ALL_REDUCE = 1
    REDUCE_SCATTER = 2


class CCParallel:
    def __init__(self, input_data, comm_type, compute_fcn,
                 compute_first=True, synchronize=True,
                 weight_shape_list=None, parallel_num=min_comm_config.parallel_num):
        self.input_data = input_data
        self.split_num = parallel_num
        self.synchronize = synchronize
        self.comm_type = comm_type
        self.compute_fcn = compute_fcn
        self.compute_first = compute_first
        self.works = []
        self.group = min_comm_config.tp_group
        self.world_size = min_comm_config.tp_world_size
        self.input_slice = input_data.shape[0] // self.split_num
        self.init_output_space(input_data, weight_shape_list, compute_first)

    def init_output_space(self, input_data, weight_shape_list, compute_first):
        if weight_shape_list is None:
            self.compute_output_shape_slice = list(input_data.shape)
        else:
            assert input_data.shape[-1] == weight_shape_list[0]
            self.compute_output_shape_slice = infer_matmul_out_shape(list(input_data.shape), weight_shape_list)
        self.output = self.allocate_output_memory()
        self.output_slice = self.output.shape[0] // self.split_num
        if compute_first:
            self.comm_output = self.output
        else:
            self.comm_output = self.allocate_communicate_memory_for_communicate_first()
        self.comm_slice = self.comm_output.shape[0] // self.split_num

    def get_dim_size_after_comm(self, dim_size):
        if self.comm_type == CommunicationType.ALL_GATHER:
            dim_size[0] = dim_size[0] * self.world_size
        elif self.comm_type == CommunicationType.REDUCE_SCATTER:
            dim_size[0] = dim_size[0] // self.world_size
        elif self.comm_type == CommunicationType.ALL_REDUCE:
            pass
        else:
            raise ValueError("Invalid comm_type.")
        return dim_size

    def allocate_output_memory(self):
        # No matter compute first or communicate first, the output shape remains the same
        output_dim_size = self.get_dim_size_after_comm(self.compute_output_shape_slice)
        output_ = torch.empty(output_dim_size, dtype=self.input_data.dtype,
                              device=torch.cuda.current_device(), requires_grad=False)
        return output_

    def allocate_communicate_memory_for_communicate_first(self):
        dim_size = list(self.input_data.shape)
        dim_size = self.get_dim_size_after_comm(dim_size)
        comm_output = torch.empty(dim_size, dtype=self.input_data.dtype,
                                  device=torch.cuda.current_device(), requires_grad=False)
        return comm_output

    def run_synchronize(self):
        for work in self.works:
            work.wait()
        return self.comm_output

    def run(self):
        if self.compute_first:
            return self.run_compute_first()
        else:
            return self.run_communicate_first()

    def comm_fcn(self, i, input_):
        if self.comm_type == CommunicationType.ALL_GATHER:
            output_ = self.comm_output[i * self.comm_slice: (i + 1) * self.comm_slice]
            work = torch.distributed._all_gather_base(output_, input_.contiguous(), group=self.group, async_op=True)
        elif self.comm_type == CommunicationType.REDUCE_SCATTER:
            output_ = self.comm_output[i * self.comm_slice: (i + 1) * self.comm_slice]
            work = torch.distributed._reduce_scatter_base(output_, input_.contiguous(), group=self.group, async_op=True)
        elif self.comm_type == CommunicationType.ALL_REDUCE:
            # all_reduce interface currently only supports overwriting the same address of input
            output_ = input_
            work = torch.distributed.all_reduce(output_, group=self.group, async_op=True)
        else:
            raise ValueError("Invalid comm_type.")
        return work, output_

    def get_input_slice(self, i):
        return self.input_data[i * self.input_slice: (i + 1) * self.input_slice]

    def run_compute_first(self):
        for i in range(self.split_num):
            input_slice = self.get_input_slice(i)
            compute_output = None
            if self.comm_type == CommunicationType.ALL_REDUCE:
                compute_output = self.output[i * self.comm_slice: (i + 1) * self.comm_slice]
                self.compute_fcn(input_tensor=input_slice, output_tensor=compute_output)
            else:
                compute_output = self.compute_fcn(input_slice)
            work, _ = self.comm_fcn(i, compute_output)
            self.works.append(work)

        if self.synchronize:
            return self.run_synchronize()
        else:
            return self.output, self.works

    def get_output_slice(self, i):
        return self.output[i * self.output_slice: (i + 1) * self.output_slice]

    def run_communicate_first(self):
        assert self.synchronize is True
        pre_work = None
        pre_output = None

        for i in range(self.split_num):
            input_slice = self.get_input_slice(i)
            if self.comm_type == CommunicationType.ALL_REDUCE:
                input_ = torch.empty_like(input_slice).copy_(input_slice)
            else:
                input_ = input_slice
            work, output_i = self.comm_fcn(i, input_)

            self.works.append(work)

            if pre_output is not None:
                pre_work.wait()
                self.compute_fcn(input_tensor=pre_output, output_tensor=self.get_output_slice(i - 1))

            pre_work = work
            pre_output = output_i

        pre_work.wait()
        self.compute_fcn(input_tensor=pre_output, output_tensor=self.get_output_slice(self.split_num - 1))
        return self.output
