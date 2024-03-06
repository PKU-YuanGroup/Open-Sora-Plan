from torch import nn
import yaml
import torch
from omegaconf import OmegaConf
from .vqgan import VQModel, GumbelVQ

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


class SDVQVAEWrapper(nn.Module):
    def __init__(self, name):
        super(SDVQVAEWrapper, self).__init__()
        raise NotImplementedError

    def encode(self, x):  # b c h w
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError
