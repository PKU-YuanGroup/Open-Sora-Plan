import torch

try:
    import torch_npu

    npu_is_available = True
except:
    npu_is_available = False


def range_push(self, msg):
    pass


def range_pop(self):
    pass


class NPUConfig:
    def __init__(self):
        self.on_npu = npu_is_available
        self.profiling = True
        self.profiling_step = 10
        if self.on_npu:
            from torch_npu.contrib import transfer_to_npu
            torch_npu.npu.set_compile_mode(jit_compile=False)

    def npu_format_cast(self, x):
        return torch_npu.npu_format_cast(x, 2)

    def run_with_dtype(self, operator, x, tmp_dtype, x_dtype):
        with torch.cuda.amp.autocast(enabled=False):
            x = operator.to(tmp_dtype)(x.to(tmp_dtype))
            return self.npu_format_cast(x).to(x_dtype)


npu_config = NPUConfig()
