import torch
import torch.nn as nn
from .block import Block
from einops import rearrange

class GroupNorm(Block):
    def __init__(self, num_channels, num_groups=32, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=True
        )
    def forward(self, x):
        return self.norm(x)

class LayerNorm(Block):
    def __init__(self, num_channels, eps=1e-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = torch.nn.LayerNorm(num_channels, eps=eps, elementwise_affine=True)
    def forward(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = self.norm(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        else:
            x = rearrange(x, "b c h w -> b h w c")
            x = self.norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x

def Normalize(in_channels, num_groups=32, norm_type="groupnorm"):
    if norm_type == "groupnorm":
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "layernorm":
        return LayerNorm(num_channels=in_channels, eps=1e-6)