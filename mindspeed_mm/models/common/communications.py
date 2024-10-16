from typing import List
import torch
import torch.distributed as dist


def _adjust_tensor_dimensions(tensor, scatter_idx, gather_idx):
    """
    Adjust the dimensions of a tensor to move scatter_idx and gather idx to dim 0 and dim 1 respectively.
    """
    dims = list(range(tensor.dim()))

    if gather_idx == 0:
        # scatter_idx >= 2
        if scatter_idx != 1:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
        # scatter_idx == 1:
        else:
            dims[scatter_idx], dims[gather_idx] = dims[gather_idx], dims[scatter_idx]
    
    elif gather_idx == 1:
        # scatter idx >= 2
        if scatter_idx != 0:
            # if scatter_idx is not 0, move it to 0
            dims[0], dims[scatter_idx] = dims[gather_idx], dims[0]
    
    # Handle the case when gather_idx >= 2
    else:
        if scatter_idx == 0:
            dims[1], dims[gather_idx] = dims[scatter_idx], dims[0]
        # scatter_idx >= 1
        else:
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]

    return tensor.permute(dims).contiguous(), dims


def _unadjust_tensor_dimensions(tensor, adjusted_dims):
    """
    Reverses the dimension adjustments using the list if adjusted dimensions.
    """
    inverse_dims = [0] * len(adjusted_dims)
    for new_pos, old_pos in enumerate(adjusted_dims):
        inverse_dims[old_pos] = new_pos
    
    # Restore the dimension order
    unadjusted_tensor = tensor.permute(inverse_dims).contiguous()
    return unadjusted_tensor


def cal_split_sizes(dim_size: int, world_size: int):
    split_size = dim_size // world_size
    remainder = dim_size % world_size
    sizes = [split_size + (1 if i < remainder else 0) for i in range(world_size)]
    return sizes


# ====================
# All-To-All
# ====================
def _all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    scatter_sizes: List = None,
    gather_sizes: List = None
):
    world_size = dist.get_world_size(group=group)

    if world_size == 1:
        return input_
    
    # 非均匀切分
    if scatter_sizes is not None and gather_sizes is not None:
        input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
        rank = dist.get_rank(group)
        output_list = []
        tensor_shape_base = input_list[rank].size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[gather_dim] = gather_sizes[i]
            output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    
    else:
        input_list = [
            t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
        ]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _single_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    scatter_sizes: List = None,
    gather_sizes: List = None
):
    sp_size = dist.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // sp_size
    if scatter_dim < 1:
        input_t = input_.reshape([sp_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1:])
    else:
        input_t = input_.reshape([-1, sp_size, inp_shape[scatter_dim]]
                                 + inp_shape[scatter_dim + 1:]).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    if scatter_dim < 1:
        output = output.transpose(0, 1).contiguous()
    return output.reshape(inp_shape[:gather_dim] + [inp_shape[gather_dim] * sp_size, ] + inp_shape[gather_dim + 1:])


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, scatter_sizes, gather_sizes, all_to_all_func):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.scatter_sizes = scatter_sizes
        ctx.gather_sizes = gather_sizes
        ctx.all_to_all_func = all_to_all_func
        output = all_to_all_func(
            input_, process_group, scatter_dim, gather_dim, scatter_sizes, gather_sizes
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = ctx.all_to_all_func(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.gather_sizes,
            ctx.scatter_sizes
        )
        return (
            grad_output,
            None,
            None,
            None,
            None,
            None,
            None
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_sizes: List = None,
    gather_sizes: List = None,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, scatter_sizes, gather_sizes, _all_to_all)


def all_to_all_SBH(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_sizes: List = None,
    gather_sizes: List = None
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, scatter_sizes, gather_sizes, _single_all_to_all)


# ====================
# Gather-Split
# ====================


def _split(input_: torch.Tensor,
    pg: dist.ProcessGroup,
    dim: int = -1,
    split_sizes: List = None
):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if world_size == 1:
        return input_

    if split_sizes is not None:
        tensor_list = torch.split(input_, split_sizes, dim=dim)
    else:
        dim_size = input_.size(dim)
        if dim_size % world_size != 0:
            raise AssertionError(
                f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), cannot split tensor evenly, please pass in the split sizes parameter"
            )
        tensor_list = torch.split(input_, dim_size // world_size, dim=dim)

    output = tensor_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor, 
    pg: dist.ProcessGroup, 
    dim: int = -1,
    gather_sizes: List = None
):
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)

    if input_.device.type not in ["cuda", "npu"]:
        raise AssertionError("input tensor must in cuda or npu")

    if world_size == 1:
        return input_

    # all gather
    if gather_sizes is not None:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, gather_sizes):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, split_sizes):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.split_sizes = split_sizes
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None


def split_forward_gather_backward(input_, process_group, dim, grad_scale=1.0, split_sizes=None):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale, split_sizes)


def gather_forward_split_backward(input_, process_group, dim, grad_scale=None, gather_sizes=None):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale, gather_sizes)