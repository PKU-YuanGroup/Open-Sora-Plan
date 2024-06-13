import ast
import os
from enum import Enum
import torch.nn.functional as F


def column_forward(self, input_, column_parallel_function=None, check_fcn=None):
    if check_fcn is not None:
        check_fcn()
    bias = self.bias if not self.skip_bias_add else None
    input_parallel = input_
    use_weight = self.weight
    if hasattr(self, "norm") and self.norm:
        use_weight = F.normalize(self.weight)
    output_parallel = column_parallel_function.apply(
        input_parallel,
        use_weight,
        bias
    )
    output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def row_forward(self, input_, row_parallel_function=None, check_fcn=None):
    if check_fcn is not None:
        check_fcn()
    input_parallel = input_
    output_parallel = row_parallel_function.apply(
        input_parallel,
        self.weight,
        None
    )
    output = output_parallel
    if not self.skip_bias_add:
        output = output + self.bias if self.bias is not None else output
        output_bias = None
    else:
        output_bias = self.bias
    return output, output_bias


class ModuleType(Enum):
    ORIGINAL_ALL_REDUCE = 0
    ORIGINAL_SEQ_PARALLEL = 1
    REWRITE_ALL_REDUCE = 2
    REWRITE_SEQ_PARALLEL = 3
    CC_FOR_ALL_REDUCE = 4
    CC_FOR_SEQ_PARALLEL = 5


class MinCommConfig:  # 由用户初始化这个类，并输入需要修改的参数，用户不感知其代码
    def __init__(self, cc_mode, cc_parallel_num):
        self.module_type: ModuleType = ModuleType.ORIGINAL_SEQ_PARALLEL  # default module_type
        self.cc_mode = cc_mode
        self.parallel_num = cc_parallel_num
        self.ColumnParallelLinear = None
        self.RowParallelLinear = None
        self.column_parallel_forward = None
        self.row_parallel_forward = None
        self.tp_group_fcn = None
        self.tp_world_size_fcn = None
        self.tp_rank_fcn = None
        self.all_reduce = None
        self.reduce_scatter_along_first_dim = None
        self.gather_along_first_dim = None
        self.prefix = None
        self.check_fcn = None

        self.sequence_parallel_enabled = True
        self.all_gather_recomputation_enabled = False
        self.print_tensor_value_enabled = False
        self.matmul_soc_friendly_enabled = True
        self.customized_cc_dict = {}
        self.tp_enabled = True
        self.k_min = 1024
        self.k_max = 4096

    def print_settings(self):
        settings_dict = {
            "cc_mode": self.cc_mode,
            "parallel_num": self.parallel_num,
            "module_type": self.module_type.name,
            "get_aligned_mm_inputs": self.matmul_soc_friendly_enabled,
            "sequence_parallel_enabled": self.sequence_parallel_enabled
        }
        print(settings_dict)

    @property
    def tp_rank(self):
        return self.tp_rank_fcn()

    @property
    def tp_group(self):
        return self.tp_group_fcn()

    @property
    def tp_world_size(self):
        return self.tp_world_size_fcn()

    def register_tp_get_functions(self, tp_group_fcn, tp_world_size_fcn, tp_rank_fcn):
        self.tp_group_fcn = tp_group_fcn
        self.tp_world_size_fcn = tp_world_size_fcn
        self.tp_rank_fcn = tp_rank_fcn

    def register_class(self, column_parallel_linear, row_parallel_linear):
        self.ColumnParallelLinear = column_parallel_linear
        self.RowParallelLinear = row_parallel_linear

    def register_mappings(self, _all_reduce, _reduce_scatter_along_first_dim, _gather_along_first_dim):
        self.all_reduce = _all_reduce
        self.reduce_scatter_along_first_dim = _reduce_scatter_along_first_dim
        self.gather_along_first_dim = _gather_along_first_dim

    def replace_forward_functions_by_autograd_class(self, column_autograd_class, row_autograd_class):
        def column_parallel_forward(x, y):
            return column_forward(x, y, column_parallel_function=column_autograd_class,
                                  check_fcn=self.check_fcn)

        def row_parallel_forward(x, y):
            return row_forward(x, y, row_parallel_function=row_autograd_class, check_fcn=self.check_fcn)

        self.column_parallel_forward = column_parallel_forward
        self.row_parallel_forward = row_parallel_forward
        self.ColumnParallelLinear.forward = self.column_parallel_forward
        self.RowParallelLinear.forward = self.row_parallel_forward

    def register_sequence_parallel_switch(self, sequence_parallel_enabled):
        self.sequence_parallel_enabled = sequence_parallel_enabled

    def register_check_fcn(self, check_fcn):
        self.check_fcn = check_fcn

    def register_customized_cc(self, customized_cc):
        if len(customized_cc) == 0:
            return
        for cc_shape_yaml_str in customized_cc.keys():
            key_list = ast.literal_eval(cc_shape_yaml_str)
            cc_shape_key_str = str(key_list)
            self.customized_cc_dict.update({cc_shape_key_str: customized_cc[cc_shape_yaml_str]})
        print("self.customized_cc_dict: ", self.customized_cc_dict)

    def register_matmul_soc_friendly_switch(self, matmul_soc_friendly):
        self.matmul_soc_friendly_enabled = matmul_soc_friendly

    def register_all_gather_recomputation_switch(self, all_gather_recomputation_enabled):
        self.all_gather_recomputation_enabled = all_gather_recomputation_enabled

    def register_print_tensor_value_switch(self, print_tensor_value_enabled):
        self.print_tensor_value_enabled = print_tensor_value_enabled

    def acquire_module_type(self, tp_size):
        sequence_parallel_types = [ModuleType.ORIGINAL_SEQ_PARALLEL,
                                   ModuleType.REWRITE_SEQ_PARALLEL,
                                   ModuleType.CC_FOR_SEQ_PARALLEL]
        all_reduce_types = [ModuleType.ORIGINAL_ALL_REDUCE,
                            ModuleType.REWRITE_ALL_REDUCE,
                            ModuleType.CC_FOR_ALL_REDUCE]

        if self.parallel_num not in [1, 2, 4, 8]:
            raise RuntimeError("CC_PARALLEL_NUM must be either 1, 2, 4 or 8. Current value not supported")
        if self.cc_mode not in [-1, 0, 1, 2]:
            raise RuntimeError("CC_MODE must be either 0, 1, or 2. Current value not supported")

        if self.cc_mode == -1:
            self.cc_mode = 0 if self.parallel_num == 1 else 2

        if tp_size == 1:
            self.cc_mode = 0
            self.parallel_num = 1

        if self.sequence_parallel_enabled:
            self.module_type = sequence_parallel_types[self.cc_mode]
        else:
            self.module_type = all_reduce_types[self.cc_mode]

        if "CC" in self.module_type.name:
            self.prefix = f"module_{self.module_type.name}_parallel_num_{self.parallel_num}"
        else:
            self.prefix = f"module_{self.module_type.name}"

        self.print_settings()


cc_mode_from_env = int(os.getenv("CC_MODE", -1))  # 0 = original, 1 = rewrite, 2 = cc default
cc_parallel_num_from_env = int(os.getenv("CC_PARALLEL_NUM", 1))

min_comm_config = MinCommConfig(cc_mode=cc_mode_from_env, cc_parallel_num=cc_parallel_num_from_env)
