import torch
import torch.nn as nn
import torch_npu
from einops import rearrange


class LayerNorm(nn.Module):
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

class NpuRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states.to(self.weight), self.weight, epsilon=self.variance_epsilon)[0]


def normalize(in_channels, num_groups=32, eps=1e-6, affine=True, norm_type="groupnorm"):
    if norm_type == "groupnorm":
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine
        )
    elif norm_type == "layernorm":
        return LayerNorm(num_channels=in_channels, eps=eps)
    elif norm_type == "rmsnorm":
        return NpuRMSNorm(hidden_size=in_channels, eps=eps)
    else:
        raise ValueError(f"unsupported norm type: {norm_type}. ")