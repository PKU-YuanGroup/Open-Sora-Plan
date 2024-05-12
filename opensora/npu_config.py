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
except:
    npu_is_available = False


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

        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch_npu.npu.set_device(self.get_local_rank())
        else:
            self.rank = torch.cuda.current_device()
            self.world_size = self.N_NPU_PER_NODE
        self.print_with_rank(f"The npu_config.on_npu is {self.on_npu}")

    def run_fa(self, query, key, value, head_dim, head_num, attention_mask=None, layout='BSH', out_dtype=None):
        if self.enable_FP32 or query.dtype == torch.float32:
            if out_dtype is None:
                out_dtype = query.dtype
            hidden_states = torch_npu.npu_fusion_attention(query.to(torch.bfloat16), key.to(torch.bfloat16),
                                                           value.to(torch.bfloat16),
                                                           atten_mask=attention_mask, input_layout=layout,
                                                           scale=1 / math.sqrt(head_dim),
                                                           head_num=head_num)[0]
            hidden_states = hidden_states.to(out_dtype)
        else:
            hidden_states = torch_npu.npu_fusion_attention(query, key, value,
                                                           atten_mask=attention_mask, input_layout=layout,
                                                           scale=1 / math.sqrt(head_dim),
                                                           head_num=head_num)[0]

        return hidden_states

    def get_output_video_path(self, name):
        os.makedirs(f"{self.work_path}/output_videos", exist_ok=True)
        return f"{self.work_path}/output_videos/{name}"

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
