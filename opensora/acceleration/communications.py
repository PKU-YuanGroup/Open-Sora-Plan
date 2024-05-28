import torch
import torch.distributed as dist
import torch.nn.functional as F
from opensora.acceleration.parallel_states import hccl_info, lccl_info, enable_LCCL
from opensora.npu_config import npu_config

try:
    from lcalib.functional import lcal_all2allvc
except:
    lcal_all2allvc = None

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


def broadcast(input_: torch.Tensor):
    if enable_LCCL:
        sp_size = lccl_info.world_size
    else:
        sp_size = hccl_info.world_size

    src = npu_config.rank // sp_size * sp_size
    # npu_config.print_msg(f"Start broadcasting, src rank is {src}...")
    dist.broadcast(input_, src=src, group=hccl_info.group)
    # npu_config.print_msg(f"Finished broadcasting, src rank is {src}...")


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
        lcal_all2allvc(input_t, output, matrix_count, lccl_info._COMM_WORLD)
    else:
        dist.all_to_all_single(output, input_t, group=hccl_info.group)
    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_dim < 1:
        output = output.transpose(0, 1).contiguous()

    output = output.reshape(
        inp_shape[: gather_dim] + [inp_shape[gather_dim] * sp_size, ] + inp_shape[gather_dim + 1:])
    return output


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


def prepare_parallel_data(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, use_image_num):
    def all_to_all(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
        hidden_states = _single_all_to_all(hidden_states, scatter_dim=2, gather_dim=0, enable_HCCL=True)
        encoder_hidden_states = _single_all_to_all(encoder_hidden_states, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        attention_mask = _single_all_to_all(attention_mask, scatter_dim=1, gather_dim=0, enable_HCCL=True)
        encoder_attention_mask = _single_all_to_all(encoder_attention_mask, scatter_dim=1, gather_dim=0,
                                                    enable_HCCL=True)
        return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask

    if use_image_num == 0:
        hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask = all_to_all(hidden_states,
                                                                                                  encoder_hidden_states,
                                                                                                  attention_mask,
                                                                                                  encoder_attention_mask)
    else:
        if enable_LCCL:
            sp_size = lccl_info.world_size
        else:
            sp_size = hccl_info.world_size
        video_states, image_states = hidden_states[:, :, :-use_image_num], hidden_states[:, :, -use_image_num:]
        video_encoder_states, image_encoder_states = encoder_hidden_states[:, :-use_image_num], encoder_hidden_states[:,
                                                                                                -use_image_num:]
        video_attention_mask, image_attention_mask = attention_mask[:, :-use_image_num], attention_mask[:,
                                                                                         -use_image_num:]
        video_encoder_attention_mask, image_encoder_attention_mask = encoder_attention_mask[:,
                                                                     :-use_image_num], encoder_attention_mask[:,
                                                                                       -use_image_num:]
        padding_needed = (sp_size - video_states.size(2) % sp_size) % sp_size
        if padding_needed > 0:
            print("Doing video padding")
            # B, C, T, H, W -> B, C, T', H, W
            video_states = F.pad(video_states, (0, 0, 0, 0, 0, padding_needed), mode='constant', value=0)
            # B, T, H, W -> B, T', H, W
            video_attention_mask = F.pad(video_attention_mask, (0, 0, 0, 0, 0, padding_needed), mode='constant',
                                         value=0)

        # mask: 1, T', H, W -> 8, T'/8, H, W
        video_states, video_encoder_states, video_attention_mask, video_encoder_attention_mask = all_to_all(
            video_states,
            video_encoder_states.repeat(1, sp_size, 1, 1),
            video_attention_mask,
            video_encoder_attention_mask.repeat(1, sp_size, 1))
        padding_needed = (sp_size - image_states.size(2) % sp_size) % sp_size
        if padding_needed > 0:
            print("Doing image padding")
            image_states = F.pad(image_states, (0, 0, 0, 0, 0, padding_needed), mode='constant', value=0)
            image_encoder_states = F.pad(image_encoder_states, (0, 0, 0, 0, 0, padding_needed), mode='constant',
                                         value=0)
            image_attention_mask = F.pad(image_attention_mask, (0, 0, 0, 0, 0, padding_needed), mode='constant',
                                         value=0)
            image_encoder_attention_mask = F.pad(image_encoder_attention_mask, (0, 0, 0, padding_needed),
                                                 mode='constant',
                                                 value=0)

        image_states, image_encoder_states, image_attention_mask, image_encoder_attention_mask = all_to_all(
            image_states,
            image_encoder_states,
            image_attention_mask,
            image_encoder_attention_mask)
        hidden_states = torch.cat([video_states, image_states], dim=2)
        encoder_hidden_states = torch.cat([video_encoder_states, image_encoder_states], dim=1)
        attention_mask = torch.cat([video_attention_mask, image_attention_mask], dim=1)
        encoder_attention_mask = torch.cat([video_encoder_attention_mask, image_encoder_attention_mask], dim=1)
        use_image_num = (use_image_num + sp_size - 1) // sp_size

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, use_image_num
