# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    import curope as _kernels # run `python setup.py install`
except ModuleNotFoundError:
    from . import curope as _kernels # run `python setup.py build_ext --inplace`


class cuRoPE3D_func (torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        # tokens = tokens.clone() # uncomment this if inplace doesn't work
        _kernels.rope_3d( tokens, positions, base, F0 )
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        _kernels.rope_3d( grad_res, positions, base, -F0 )
        ctx.mark_dirty(grad_res)
        return grad_res, None, None, None


class cuRoPE3D(torch.nn.Module):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=None):
        super().__init__()
        self.base = freq 
        self.F0 = F0

    def forward(self, tokens, positions): 
        # tokens.transpose(1,2): B,N,H,D
        # positions: B,N,3
        # print('tokens.transpose(1,2).shape, positions.shape, self.base, self.F0', 
        #       tokens.transpose(1,2).shape, positions.shape, self.base, self.F0)
        cuRoPE3D_func.apply( tokens.transpose(1,2), positions, self.base, self.F0 )
        return tokens