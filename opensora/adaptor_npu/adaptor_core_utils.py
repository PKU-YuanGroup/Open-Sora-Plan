import torch
import opensora.core


def _kernel_make_viewless_tensor(inp, requires_grad):
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad, )
    with torch.no_grad():
        out.set_(inp.data)
    return out


opensora.core.utils._kernel_make_viewless_tensor = _kernel_make_viewless_tensor
