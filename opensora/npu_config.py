import os
import torch

try:
    import torch_npu
    npu_is_available = True
except:
    npu_is_available = False


class NPUConfig:
    def __init__(self):
        self.on_npu = npu_is_available
        self.profiling = True
        self.profiling_step = 5
        self._loss = []
        self.rank = int(os.environ["RANK"])
        if self.on_npu:
            from torch_npu.contrib import transfer_to_npu
            torch_npu.npu.set_compile_mode(jit_compile=False)

        self.print_with_rank(f"The npu_config.on_npu is {self.on_npu}")

    def npu_format_cast(self, x):
        return torch_npu.npu_format_cast(x, 2)

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

    def run_conv3d(self, operator, x, out_dtype):
        return self._run(operator, x, torch.float16, out_dtype, out_nd_format=True)

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