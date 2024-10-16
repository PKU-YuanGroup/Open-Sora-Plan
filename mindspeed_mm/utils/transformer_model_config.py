import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig

from mindspeed_mm.configs.config import ConfigReader
from .utils import get_dtype, quick_gelu


def get_class_variables(cls):
    all_members = dir(cls)
    filtered_members = [member for member in all_members if not member.startswith("__")]

    return filtered_members


def get_model_config(config):
    config_dict = config.to_dict()
    t_config = dict()
    tfc_variables = get_class_variables(TransformerConfig)
    for key in tfc_variables:
        if key in config_dict.keys():
            t_config[key] = config_dict[key]
    t_config["params_dtype"] = get_dtype(t_config.get("params_dtype"))
    if t_config.get("activation_func") == "silu":
        t_config["activation_func"] = F.silu
    elif t_config.get("activation_func") == "quick_gelu":
        t_config["activation_func"] = quick_gelu
    else:
        t_config["activation_func"] = F.gelu
    
    trans_config = TransformerConfig(**t_config)

    for key in tfc_variables:
        config_dict[key] = getattr(trans_config, key)
    new_config = ConfigReader(config_dict)

    return new_config