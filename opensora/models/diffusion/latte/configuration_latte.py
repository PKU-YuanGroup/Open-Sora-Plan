import json
from typing import Union, Tuple


class LatteConfiguration:

    def __init__(
        self,
        input_size=(32, 32),
        patch_size=(2, 2),
        patch_size_t=1,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=1,
        attention_mode='math',
        compress_kv=False,
        attention_pe_mode=None,
        pt_input_size: Union[int, Tuple[int, int]] = None,  # (h, w)
        pt_num_frames: Union[int, Tuple[int, int]] = None,  # (num_frames, 1)
        intp_vfreq: bool = True,  # vision position interpolation
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.extras = learn_sigma
        self.attention_mode = attention_mode
        self.compress_kv = compress_kv
        self.attention_pe_mode = attention_pe_mode
        self.pt_input_size = pt_input_size
        self.pt_num_frames = pt_num_frames
        self.intp_vfreq = intp_vfreq

    def to_json_string(self):
        json_string = json.dumps(vars(self))
        return json_string

    def to_dict(self):
        return vars(self)

    @classmethod
    def load_from_file(cls, config_path):
        with open(config_path, 'r') as json_file:
            config_dict = json.load(json_file)
        return cls(**config_dict)


def Latte_XL_122_Config(**kwargs):
    return LatteConfiguration(depth=56, hidden_size=1152, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def Latte_XL_144_Config(**kwargs):
    return LatteConfiguration(depth=56, hidden_size=1152, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def Latte_XL_188_Config(**kwargs):
    return LatteConfiguration(depth=56, hidden_size=1152, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def Latte_L_122_Config(**kwargs):
    return LatteConfiguration(depth=48, hidden_size=1024, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def Latte_L_144_Config(**kwargs):
    return LatteConfiguration(depth=48, hidden_size=1024, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def Latte_L_188_Config(**kwargs):
    return LatteConfiguration(depth=48, hidden_size=1024, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def Latte_B_122_Config(**kwargs):
    return LatteConfiguration(depth=24, hidden_size=768, patch_size_t=1, patch_size=2, num_heads=12, **kwargs)

def Latte_B_144_Config(**kwargs):
    return LatteConfiguration(depth=24, hidden_size=768, patch_size_t=1, patch_size=4, num_heads=12, **kwargs)

def Latte_B_188_Config(**kwargs):
    return LatteConfiguration(depth=24, hidden_size=768, patch_size_t=1, patch_size=8, num_heads=12, **kwargs)

def Latte_S_122_Config(**kwargs):
    return LatteConfiguration(depth=24, hidden_size=384, patch_size_t=1, patch_size=2, num_heads=6, **kwargs)

def Latte_S_144_Config(**kwargs):
    return LatteConfiguration(depth=24, hidden_size=384, patch_size_t=1, patch_size=4, num_heads=6, **kwargs)

def Latte_S_188_Config(**kwargs):
    return LatteConfiguration(depth=24, hidden_size=384, patch_size_t=1, patch_size=8, num_heads=6, **kwargs)


Latte_configs = {
    "Latte-XL/122": Latte_XL_122_Config, "Latte-XL/144": Latte_XL_144_Config, "Latte-XL/188": Latte_XL_188_Config,
    "Latte-L/122": Latte_L_122_Config, "Latte-L/144": Latte_L_144_Config, "Latte-L/188": Latte_L_188_Config,
    "Latte-B/122": Latte_B_122_Config, "Latte-B/144": Latte_B_144_Config, "Latte-B/188": Latte_B_188_Config,
    "Latte-S/122": Latte_S_122_Config, "Latte-S/144": Latte_S_144_Config, "Latte-S/188": Latte_S_188_Config,
}
