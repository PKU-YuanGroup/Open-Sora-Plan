# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from .distributed import DistributedDataParallel
from .module import Float16Module

from torch.nn.parallel import DistributedDataParallel as torchDDP
def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model
