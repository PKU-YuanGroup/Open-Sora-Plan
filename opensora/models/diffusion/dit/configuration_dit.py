import json
from typing import Union, Tuple


class DiTConfiguration:

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        patch_size_t=1,
        in_channels=256,
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
        attention_pe_mode=None,
        pt_input_size: Union[int, Tuple[int, int]] = None,  # (h, w)
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
        self.attention_pe_mode = attention_pe_mode
        self.pt_input_size = pt_input_size
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


def DiT_XL_122_Config(**kwargs):
    return DiTConfiguration(depth=28, hidden_size=1152, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_144_Config(**kwargs):
    return DiTConfiguration(depth=28, hidden_size=1152, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_188_Config(**kwargs):
    return DiTConfiguration(depth=28, hidden_size=1152, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def DiT_L_122_Config(**kwargs):
    return DiTConfiguration(depth=24, hidden_size=1024, patch_size_t=1, patch_size=2, num_heads=16, **kwargs)

def DiT_L_144_Config(**kwargs):
    return DiTConfiguration(depth=24, hidden_size=1024, patch_size_t=1, patch_size=4, num_heads=16, **kwargs)

def DiT_L_188_Config(**kwargs):
    return DiTConfiguration(depth=24, hidden_size=1024, patch_size_t=1, patch_size=8, num_heads=16, **kwargs)

def DiT_B_122_Config(**kwargs):
    return DiTConfiguration(depth=12, hidden_size=768, patch_size_t=1, patch_size=2, num_heads=12, **kwargs)

def DiT_B_144_Config(**kwargs):
    return DiTConfiguration(depth=12, hidden_size=768, patch_size_t=1, patch_size=4, num_heads=12, **kwargs)

def DiT_B_188_Config(**kwargs):
    return DiTConfiguration(depth=12, hidden_size=768, patch_size_t=1, patch_size=8, num_heads=12, **kwargs)

def DiT_S_122_Config(**kwargs):
    return DiTConfiguration(depth=12, hidden_size=384, patch_size_t=1, patch_size=2, num_heads=6, **kwargs)

def DiT_S_144_Config(**kwargs):
    return DiTConfiguration(depth=12, hidden_size=384, patch_size_t=1, patch_size=4, num_heads=6, **kwargs)

def DiT_S_188_Config(**kwargs):
    return DiTConfiguration(depth=12, hidden_size=384, patch_size_t=1, patch_size=8, num_heads=6, **kwargs)


DiT_configs = {
    'DiT-XL/122': DiT_XL_122_Config,  'DiT-XL/144': DiT_XL_144_Config,  'DiT-XL/188': DiT_XL_188_Config,
    'DiT-L/122':  DiT_L_122_Config,   'DiT-L/144':  DiT_L_144_Config,   'DiT-L/188':  DiT_L_188_Config,
    'DiT-B/122':  DiT_B_122_Config,   'DiT-B/144':  DiT_B_144_Config,   'DiT-B/188':  DiT_B_188_Config,
    'DiT-S/122':  DiT_S_122_Config,   'DiT-S/144':  DiT_S_144_Config,   'DiT-S/188':  DiT_S_188_Config,
}
