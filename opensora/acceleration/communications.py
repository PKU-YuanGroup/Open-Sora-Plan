import torch
import torch.distributed as dist
from einops import rearrange
from opensora.acceleration.parallel_states import hccl_info, lccl_info, enable_LCCL
try:
    from lcalib.functional import lcal_all2allvc
except:
    lcal_all2allvc = None

def broadcast(input_: torch.Tensor):
    sp_size = hccl_info.world_size
    src = hccl_info.rank // sp_size * sp_size
    dist.broadcast(input_, src=src, group=hccl_info.group)

_COUNT = 0
def _all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
):
    group = hccl_info.group
    sp_size = hccl_info.world_size
    input_list = [t.contiguous() for t in torch.tensor_split(input_, sp_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(sp_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()

def _single_all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    enable_HCCL=False,
):
    if enable_LCCL:
        sp_size = lccl_info.world_size
    else:
        sp_size = hccl_info.world_size
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // sp_size
    if scatter_dim < 1:
        input_t = input_.reshape(
            [sp_size, inp_shape[scatter_dim]] + \
            inp_shape[scatter_dim + 1:]
        )
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input_.reshape(
            [-1, sp_size, inp_shape[scatter_dim]] + \
            inp_shape[scatter_dim + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    if enable_LCCL and not enable_HCCL:
        matrix_count = torch.ones([sp_size, sp_size], dtype=torch.int64, device=input_t.device) * (
                    input_t.numel() // sp_size)
        lcal_all2allvc(input_t, output, matrix_count, lccl_info.group)
    else:
        dist.all_to_all_single(output, input_t, group=hccl_info.group)
    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_dim < 1:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_dim] + [inp_shape[gather_dim] * sp_size, ] + inp_shape[gather_dim + 1:])


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim, all_to_all_func):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.all_to_all = all_to_all_func
        output = ctx.all_to_all(input_, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = ctx.all_to_all(
            grad_output,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )

def all_to_all_SBH(
    input_: torch.Tensor,
    scatter_dim: int = 1,
    gather_dim: int = 0,
):
    return _AllToAll.apply(input_, scatter_dim, gather_dim, _single_all_to_all)

def all_to_all_BSND(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, scatter_dim, gather_dim, _all_to_all)


def prepare_parallel_data(
        hidden_states, 
        encoder_hidden_states, 
        attention_mask, 
        encoder_attention_mask, 
        pooled_projections=None, 
        ):
    def all_to_all(
            hidden_states, 
            encoder_hidden_states, 
            attention_mask, 
            encoder_attention_mask, 
            pooled_projections, 
            ):
        # hidden_states          (b c t h w)   -gather0-> (sp*b c t h w)   -scatter2-> (sp*b c t//sp h w)
        # encoder_hidden_states  (b sp l/sp d) -gather0-> (sp*b sp l/sp d) -scatter1-> (sp*b 1 l/sp d)
        # attention_mask         (b t*sp h w)  -gather0-> (sp*b t*sp h w)  -scatter1-> (sp*b t h w)
        # encoder_attention_mask (b sp l)      -gather0-> (sp*b sp l)      -scatter1-> (sp*b 1 l)
        # pooled_projections     (b sp d)      -gather0-> (sp*b sp d)      -scatter1-> (sp*b 1 d)
        hidden_states = _single_all_to_all(hidden_states, scatter_dim=2, gather_dim=0, enable_HCCL=True)
        encoder_hidden_states = _single_all_to_all(encoder_hidden_states, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        attention_mask = _single_all_to_all(attention_mask, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        encoder_attention_mask = _single_all_to_all(encoder_attention_mask, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        if pooled_projections is not None:
            pooled_projections = _single_all_to_all(pooled_projections, scatter_dim=1, gather_dim=0, enable_HCCL=True)

        return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, pooled_projections

    sp_size = hccl_info.world_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    encoder_hidden_states = rearrange(
        encoder_hidden_states, 'b 1 (n x) h -> b n x h',
        n=sp_size, x=encoder_hidden_states.shape[2]//sp_size
        ).contiguous()
    hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, pooled_projections = all_to_all(
        hidden_states, 
        encoder_hidden_states, 
        attention_mask.repeat(1, sp_size, 1, 1), 
        encoder_attention_mask.repeat(1, sp_size, 1), 
        pooled_projections.repeat(1, sp_size, 1)
        )

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, pooled_projections