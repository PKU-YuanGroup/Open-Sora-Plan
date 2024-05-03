import os
import pickle

import torch

try:
    import torch_npu

    npu_is_available = True
except:
    npu_is_available = False


class NPUConfig:
    N_NPU_PER_NODE = 8

    def __init__(self):
        self.on_npu = npu_is_available
        self.node_world_size = 8
        self.profiling = False
        self.profiling_step = 5
        self._loss = []
        self.enable_FA = False
        self.work_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pickle_save_path = f"{self.work_path}/pickles"
        self.load_pickle = True

        if self.on_npu:
            from torch_npu.contrib import transfer_to_npu
            torch_npu.npu.set_compile_mode(jit_compile=True)

        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        else:
            self.rank = torch.cuda.current_device()
        self.print_with_rank(f"The npu_config.on_npu is {self.on_npu}")

    def get_pickle_path(self, file_name):
        return f"{self.pickle_save_path}/{file_name}_{self.rank % self.N_NPU_PER_NODE}.pkl"

    def try_load_pickle(self, file_name, function):
        file_name = self.get_pickle_path(file_name)
        if os.path.exists(file_name) and self.load_pickle:
            with open(file_name, 'rb') as file:
                loaded_data = pickle.load(file)
                return loaded_data
        else:
            data = function()
            os.makedirs(self.pickle_save_path, exist_ok=True)
            with open(file_name, 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            return data

    def npu_format_cast(self, x):
        return torch_npu.npu_format_cast(x, 2)

    def print_grad_norm(self, model):
        # 计算并打印梯度范数
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (1. / 2)
        self.print_msg(f'Gradient Norm is : {grad_norm}')

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
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        mean_val = tensor.mean().item()
        median_val = tensor.median().item()
        std_val = tensor.std().item()
        shape = tensor.shape
        self.print_msg(
            f"{name} - Max: {max_val}, Min: {min_val}, Mean: {mean_val}, Median: {median_val}, Std: {std_val}, Shape: {shape}")

    def run_conv3d(self, operator, x, out_dtype):
        return self._run(operator, x, torch.float32, out_dtype, out_nd_format=True)

    def run_pad3d(self, operator, x):
        n, c, d, h, w = x.shape
        x = x.reshape(n * c, d, h * w)
        x = operator(x)
        x = x.reshape(n, c, -1, h, w)
        return x

    def seed_everything(seed=0):
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

    def print_msg(self, msg, on=True):
        if on:
            print(f"[RANK-{self.rank}]: {msg}", flush=True)

    def save_loss(self, filename, rank=0):
        if self.rank == rank:
            import json
            with open(filename, 'w') as file:
                json.dump(self._loss, file, indent=4)

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
