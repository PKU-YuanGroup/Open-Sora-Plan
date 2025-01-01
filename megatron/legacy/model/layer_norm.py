import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):

    def __init__(self,
                dim: int,
                eps: float = 1e-6,
                sequence_parallel: bool = False):
        super().__init__()
        self.dim = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim))
        self.bias = nn.Parameter(torch.zeros(self.dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)
        setattr(self.bias, 'sequence_parallel', sequence_parallel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.dim,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)      