import os
import json

from mindspeed_mm.utils.utils import get_dtype


class ConfigReader:
    """  
    read_config read json file dict processed by MMconfig
    and convert to class attributes, besides, read_config
    support to convert dict for specific purposes.
    """
    def __init__(self, config_dict: dict) -> None:
        for k, v in config_dict.items():
            if k == "dtype":
                v = get_dtype(v)
            if isinstance(v, dict):
                self.__dict__[k] = ConfigReader(v)
            else:
                self.__dict__[k] = v
    
    def to_dict(self) -> dict:
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret
    
    def __repr__(self) -> str:
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                print(">>>>> {}".format(k))
                print(v)
            else:
                print("{}: {}".format(k, v))
        return ""

    def __str__(self) -> str:
        self.__repr__()
        return ""

    def update_unuse(self, **kwargs):

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


class MMConfig:
    """ 
    MMconfig 
        input: a dict of json path
    """
    def __init__(self, json_files: dict) -> None:
        for json_name, json_path in json_files.items():
            if os.path.exists(json_path):
                real_path = os.path.realpath(json_path)
                config_dict = self.read_json(real_path)
                setattr(self, json_name, ConfigReader(config_dict))
    
    def read_json(self, json_path):
        with open(json_path, mode="r") as f:
            json_file = f.read()
        config_dict = json.loads(json_file)
        return config_dict
   
    
def _add_mm_args(parser):
    group = parser.add_argument_group(title="multimodel")
    group.add_argument("--model_custom_precision", action="store_true", default=True, help="Use custom precision for model, e.g., we use fp32 for vae and fp16 for predictor.")
    group.add_argument("--clip_grad_ema_decay", type=float, default=0.99, help="EMA decay coefficient of Adaptive Gradient clipping in Open-Sora Plan based on global L2 norm.")
    group.add_argument("--selective_recom", action="store_true", default=False, help="Use selective recomputation in Open-Sora Plan.")
    group.add_argument("--recom_ffn_layers", type=int, default=32, help="Number of FFN layers when we use selective recomputation in Open-Sora Plan.")
    group.add_argument("--mm-data", type=str, default="")
    group.add_argument("--mm-model", type=str, default="")
    group.add_argument("--mm-tool", type=str, default="")
    return parser


def mm_extra_args_provider(parser):
    parser = _add_mm_args(parser)
    return parser


def merge_mm_args(args):
    setattr(args, "mm", object)
    json_files = {"model": args.mm_model, "data": args.mm_data, "tool": args.mm_tool}
    args.mm = MMConfig(json_files)
    args.curr_forward_iteration = 0
    