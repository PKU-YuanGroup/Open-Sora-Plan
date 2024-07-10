import torch
from torch import nn
from torch.nn import functional as F


def fp32_layer_norm_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    origin_dtype = inputs.dtype
    return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None,
                        self.bias.float() if self.bias is not None else None, self.eps).to(origin_dtype)


def fp32_silu_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(inputs.float(), inplace=self.inplace).to(inputs.dtype)


def fp32_gelu_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(inputs.float(), approximate=self.approximate).to(inputs.dtype)


def replace_with_fp32_forwards():
    nn.GELU.forward = fp32_gelu_forward
    nn.SiLU.forward = fp32_silu_forward
    nn.LayerNorm.forward = fp32_layer_norm_forward
