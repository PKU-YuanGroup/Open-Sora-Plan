import torch
import torch.distributed as dist


# ====================
# All-To-All
# ====================
def _all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    world_size = dist.get_world_size(group=group)
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
    gather_dim: int
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
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, all_to_all_func):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.all_to_all_func = all_to_all_func
        output = all_to_all_func(
            input_, process_group, scatter_dim, gather_dim
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = ctx.all_to_all_func(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, _all_to_all)


def all_to_all_SBH(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, _single_all_to_all)


# ====================
# Gather-Split
# ====================


def _split(input_, pg: dist.ProcessGroup, dim=-1):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    if dim_size % world_size != 0:
        raise AssertionError(
            f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), cannot split tensor evenly"
        )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: dist.ProcessGroup, dim=-1):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    if input_.device.type not in ["cuda", "npu"]:
        raise AssertionError
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
    def forward(ctx, input_, process_group, dim, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim), None, None, None


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
    def forward(ctx, input_, process_group, dim, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim), None, None, None


def split_forward_gather_backward(input_, process_group, dim, grad_scale=1.0):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)


def gather_forward_split_backward(input_, process_group, dim, grad_scale=None):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)
