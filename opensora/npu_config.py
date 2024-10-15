import math
import mmap
import os
import pickle
import random
import numpy as np
import torch
import subprocess
import sys
import threading
import gc
import torch.distributed as dist

from opensora.adaptor.zp_manager import zp_manager

try:
    import torch_npu

    npu_is_available = True
    from torch_npu.contrib import transfer_to_npu
except:
    npu_is_available = False

from contextlib import contextmanager
import types


def compress_video(input_file, output_file, out_size):
    """使用 ffmpeg 压缩视频文件。"""
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f"scale='min({out_size},iw)':'min({out_size},ih)':force_original_aspect_ratio=decrease",
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'slow',
        '-c:a', 'copy',
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@contextmanager
def set_run_dtype(x, dtype=None):
    # 保存原始环境变量的值（如果存在）
    npu_config.original_run_dtype = x.dtype
    # 设置环境变量为指定的值
    npu_config.current_run_dtype = dtype
    try:
        # Yield control back to the body of the `with` statement
        yield
    finally:
        # 恢复原始的环境变量值
        npu_config.current_run_dtype = None
        npu_config.original_run_dtype = None


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
        self.current_run_dtype = None
        self.original_run_dtype = None
        self.zp_manager = zp_manager
        self.replaced_type = torch.float32
        self.conv_dtype = torch.float16
        if self.enable_FA and self.enable_FP32:
            self.inf_float = -10000.0
        else:
            self.inf_float = -10000.0

        if self.use_small_dataset:
            self.load_pickle = False

        self._loss = []
        self.work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pickle_save_path = f"{self.work_path}/pickles"
        self.mm = dict()

        if self.on_npu:
            import deepspeed
            import sys
            torch_npu.npu.set_compile_mode(jit_compile=False)

            import deepspeed.runtime.utils as utils
            from opensora.adaptor.utils import all_gather_dp_groups, all_gather_into_tensor_dp_groups
            utils.all_gather_dp_groups = all_gather_dp_groups

            import deepspeed.runtime.bf16_optimizer as bf16_optimizer
            from opensora.adaptor.bf16_optimizer import BF16_Optimizer
            self.replace_methods(bf16_optimizer.BF16_Optimizer, BF16_Optimizer)

            from opensora.adaptor.stage_1_and_2 import DeepSpeedZeroOptimizer
            import deepspeed.runtime.zero.stage_1_and_2 as stage_1_and_2
            self.replace_methods(stage_1_and_2.DeepSpeedZeroOptimizer, DeepSpeedZeroOptimizer, ['_has_inf_or_nan'])

            import deepspeed.runtime.engine as engine
            from opensora.adaptor.engine import DeepSpeedEngine
            self.replace_methods(engine.DeepSpeedEngine, DeepSpeedEngine, skip_fcns=['__init__', '_copy_recovery_script', '_change_recovery_script_permissions'])

        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch_npu.npu.set_device(self.get_local_rank())
        else:
            self.rank = torch.cuda.current_device()
            self.world_size = self.N_NPU_PER_NODE
        self.print_with_rank(f"The npu_config.on_npu is {self.on_npu}")
        self.bind_thread_to_cpu()
        gc.set_threshold(700, 10, 10000)

    def get_total_cores(self):
        try:
            total_cores = os.sysconf('SC_NPROCESSORS_ONLN')
        except (AttributeError, ValueError):
            total_cores = os.cpu_count()
        return total_cores


    def bind_thread_to_cpu(self):
        total_cores = self.get_total_cores()
        # 每个卡的核心数量
        cores_per_rank = total_cores // 8
        # 计算本地rank
        local_rank = self.rank % 8
        # 计算当前 rank 的 CPU 核范围
        start_core = local_rank * cores_per_rank
        end_core = start_core + cores_per_rank - 1
        # 构建 CPU 核范围字符串
        cpu_cores_range = f"{start_core}-{end_core}"
        pid = os.getpid()
        command = f"taskset -cp {cpu_cores_range} {pid}"

        subprocess.run(command, shell=True, check=True)
        return f"Binding Cores:{self.rank}:{pid}:{cpu_cores_range}"

    def replace_methods(self, target_class, source_class, skip_fcns=[], only_include_fcns=None):
        for attr_name in dir(source_class):
            attr_value = getattr(source_class, attr_name)
            if attr_name in source_class.__dict__:
                attr_class_value = source_class.__dict__[attr_name]
            else:
                attr_class_value = attr_value
            if (isinstance(attr_class_value, staticmethod) or isinstance(attr_class_value, classmethod)
                    or attr_name in skip_fcns):
                print(f"skip replace {attr_name}")
                continue

            if only_include_fcns is not None and attr_name not in only_include_fcns:
                continue

            elif isinstance(attr_value, types.FunctionType):
                setattr(target_class, attr_name, attr_value)

    def get_attention_mask(self, attention_mask, repeat_num):
        if self.on_npu and attention_mask is not None:
            if npu_config.enable_FA:
                attention_mask = attention_mask.to(torch.bool)
            attention_mask = attention_mask.repeat_interleave(repeat_num, dim=-2)
        return attention_mask
    def set_current_run_dtype(self, variables):
        if variables[0].dtype != self.current_run_dtype and self.current_run_dtype is not None:
            for index, var in enumerate(variables):
                variables[index] = var.to(self.current_run_dtype)
        return tuple(variables)

    def restore_dtype(self, x):
        if x.dtype != self.original_run_dtype and self.original_run_dtype is not None:
            x = x.to(self.original_run_dtype)
        return x

    def get_output_video_path(self, name):
        os.makedirs(f"{self.work_path}/output_videos", exist_ok=True)
        return f"{self.work_path}/output_videos/{name}"

    def get_node_id(self):
        return self.rank // self.node_world_size

    def get_node_size(self):
        return self.world_size // self.node_world_size

    def get_local_rank(self):
        return self.rank % self.N_NPU_PER_NODE

    def get_pickle_path(self, file_name):
        return f"{self.pickle_save_path}/{file_name}_local_n63"

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

    def try_get_vid_path(self, file, out_size=1024):
        output_file = file.rsplit(".", 1)[0] + f"_resize{out_size}.mp4"
        if not os.path.exists(output_file):
            return file
        #     compress_video(file, output_file, out_size)
        return output_file

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
        # for name, param in model.named_parameters():
        #     grad_data = deepspeed.utils.safe_get_full_grad(param)
        #     # self.print_tensor_stats(grad_data, name=name)
        #
        #     if grad_data is not None:
        #         param_norm = grad_data.norm(2)
        #         grad_norm += param_norm.item() ** 2
        #         n_grad += 1
        # grad_norm = (grad_norm / n_grad) ** (1. / 2)

        return grad_norm

    def _run(self, operator, x, tmp_dtype, out_dtype=None, out_nd_format=False):
        if self.on_npu:
            if out_dtype is None:
                out_dtype = x.dtype

            with torch.cuda.amp.autocast(enabled=False):
                x = operator.to(device=x.device, dtype=tmp_dtype)(x.to(tmp_dtype))
                x = x.to(out_dtype)
                if out_nd_format:
                    return self.npu_format_cast(x)
                else:
                    return x
        else:
            return operator(x)

    def run_group_norm(self, operator, x):
        return self._run(operator, x, torch.float32)

    def run_layer_norm(self, operator, x):
        return self._run(operator, x, torch.float32)

    def print_tensor_stats(self, tensor, name="Tensor", rank=None):
        if rank and rank != self.rank:
            return

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
        return self._run(operator, x, self.conv_dtype, out_dtype, out_nd_format=True)

    def run_pool_2d(self, operator, x):
        return self._run(operator, x, self.replaced_type)

    def run_pad_2d(self, operator, x, pad, mode="constant"):
        if self.on_npu:
            x_dtype = x.dtype
            x = x.to(self.replaced_type)
            x = operator(x, pad, mode)
            x = x.to(x_dtype)
        else:
            x = operator(x, pad, mode)
        return x

    def seed_everything(self, seed=100):
        seed += self.rank
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

    def run_attention(self, query, key, value, atten_mask, input_layout, head_dim, head_num):
        if self.enable_FA:
            hidden_states = torch_npu.npu_fusion_attention(query, key, value,
                                                           atten_mask=atten_mask,
                                                           input_layout=input_layout,
                                                           scale=1 / math.sqrt(head_dim),
                                                           head_num=head_num)[0]
        else:
            hidden_states = self.scaled_dot_product_attention(query, key, value,
                                                              atten_mask=atten_mask,
                                                              input_layout=input_layout,
                                                              scale=1 / math.sqrt(head_dim),
                                                              head_num=head_num)
        return hidden_states

    def scaled_dot_product_attention(self, query, key, value, input_layout, head_num=None,
                                     atten_mask=None, scale=None, dropout_p=0.0, is_causal=False) -> torch.Tensor:
        # L, S = query.size(-2), key.size(-2)
        def trans_tensor_shape(x, layout, head_num):
            if layout == "BSH":
                batch = x.shape[0]
                x = x.view(batch, -1, head_num, x.shape[-1] // head_num).transpose(1, 2).contiguous()
            elif layout == "SBH":
                batch = x.shape[1]
                x = x.view(-1, batch * head_num, x.shape[-1] // head_num).transpose(0, 1).contiguous()
                x = x.view(batch, head_num, -1, x.shape[-1])
            return x

        query = trans_tensor_shape(query, input_layout, head_num)
        key = trans_tensor_shape(key, input_layout, head_num)
        value = trans_tensor_shape(value, input_layout, head_num)

        attn_weight = query @ key.transpose(-2, -1) * scale
        attn_bias = torch.zeros_like(attn_weight, dtype=query.dtype, device=query.device)
        if is_causal:
            assert atten_mask is None
            temp_mask = torch.zeros_like(attn_weight, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), npu_config.inf_float)
            attn_bias.to(query.dtype)

        if atten_mask is not None:
            assert (not self.enable_FA) and atten_mask.dtype != torch.bool, \
                "attention_mask must not be bool type when use this function"

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value
        if input_layout == "BSH":
            output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, head_num * output.shape[-1])
        else:
            output = output.view(output.shape[0] * head_num, -1, output.shape[-1]).transpose(0, 1).contiguous()
            output = output.view(output.shape[0], -1, head_num * output.shape[-1])
        return output

    def print_tensor_with_rank(self, name, tensor, rank=[0], dim_print_cnt=[]):
        if type(rank) is not list:
            rank = [rank]
        if self.rank in rank:
            def print_dim(tensor_, indices):
                if tensor_.dim() == len(indices):
                    return '{0:10.5f} '.format(tensor[tuple(indices)].detach().item())
                else:
                    cur_dim = len(indices)
                    ret = ''
                    for x in range(0, tensor_.size(cur_dim), tensor_.size(cur_dim) // dim_print_cnt[cur_dim]):
                        ret += print_dim(tensor_, indices + [x])
                    return ret + '\n'

            print(name, tensor.size(), self.rank, '\n', print_dim(tensor, []))


npu_config = NPUConfig()
