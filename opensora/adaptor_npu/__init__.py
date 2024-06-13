import os
import copy
import sys
import types

import torch
import torch_npu
from functools import wraps
from torch_npu.contrib import transfer_to_npu
from . import adaptor_amp_c

if 'amp_C' in sys.modules:
    del sys.modules['amp_C']
sys.modules['amp_C'] = __import__('opensora.adaptor_npu.adaptor_amp_c')

if torch.__version__ == '2.1.0':
    sys.modules['torch._six'] = types.ModuleType('torch_six')
    setattr(sys.modules['torch._six'], 'inf', torch.nn.Module)

global FLAG_SUPPORT_INF_NAN
FLAG_SUPPORT_INF_NAN = hasattr(torch_npu.npu.utils, 'is_support_inf_nan') and torch_npu.npu.utils.is_support_inf_nan()

from opensora.adaptor_npu import adaptor_core_tensor_parallel
from opensora.adaptor_npu import adaptor_core_utils
from opensora.adaptor_npu import adaptor_optimizer_clip_grads
from opensora.adaptor_npu import adaptor_optimizer_distrib_optimizer
from opensora.adaptor_npu import adaptor_optimizer_optimizer
from opensora.adaptor_npu import adaptor_core_cross_entropy
from opensora.adaptor_npu import adaptor_core_layers

def wrapper_type(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        output = fn(*args, **kwargs)
        if isinstance(output, str):
            if output == 'torch.npu.FloatTensor':
                output = 'torch.cuda.FloatTensor'
            elif output == 'torch.npu.HalfTensor':
                output = 'torch.cuda.HalfTensor'
        return output

    return decorated


# deprecated
def wrapper_dist(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if args[0].dtype == torch.long and not kwargs.get('async_op', False):
            new_args = list(copy.deepcopy(args))
            new_args[0] = new_args[0].int()
            fn(*new_args, **kwargs)
            args[0].copy_(new_args[0].long())
            return
        return fn(*args, **kwargs)

    return wrapper


os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
torch.Tensor.type = wrapper_type(torch.Tensor.type)
torch.distributed.all_reduce = wrapper_dist(torch.distributed.all_reduce)
