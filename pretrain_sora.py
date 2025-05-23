# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain SoRA."""

import torch

import mindspeed.megatron_adaptor

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_max_loss_across_data_parallel_group,
    gather_info_from_all_processes,
    unwrap_model,
)
from megatron.training.global_vars import get_wandb_writer
from megatron.core.optimizer.clip_grads import AdaptiveGradClipInfo

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    VIDEO, 
    PROMPT_IDS, 
    PROMPT_MASK, 
    VIDEO_MASK,
    PROMPT_IDS_2, 
    PROMPT_MASK_2, 
)
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.sora_model import SoRAModel


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building SoRA model ...")
    model = SoRAModel(args.mm.model)
    return model


def get_batch_on_this_tp_rank(data_iterator):
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(torch.cuda.current_device())
    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch
    else:
        return None


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor.mean()
    max_loss = get_max_loss_across_data_parallel_group([loss])
    # loss_clone = loss.clone().detach()
    # loss_clone = (1.0 - AdaptiveGradClipInfo.zero_grad_flag) * loss_clone / (AdaptiveGradClipInfo.clip_coef + 1e-15)
    # avg_loss = average_losses_across_data_parallel_group([loss_clone])
    avg_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"avg_loss": avg_loss[0], "max_loss": max_loss[0]}

def get_batch_for_step(data_iterator):
    args = get_args()
    args.curr_forward_iteration += 1
    enable_encoder_dp = args.mm.model.enable_encoder_dp if hasattr(args.mm.model, "enable_encoder_dp") else False
    tp_cp_group_size = torch.distributed.get_world_size(mpu.get_tensor_and_context_parallel_group())

    if not enable_encoder_dp or tp_cp_group_size <= 1:
        return get_batch(data_iterator)

    # Only the first step of a round needs to get batch when enable encoder dp
    batch = get_batch(data_iterator) if args.curr_forward_iteration % tp_cp_group_size == 1 else {}
    return batch

def forward_step(data_iterator, model):
    """Forward step."""
    torch.distributed.barrier()
    batch = get_batch_for_step(data_iterator)
    torch.distributed.barrier()
    video = batch.pop(VIDEO, None)
    prompt_ids = batch.pop(PROMPT_IDS, None)
    video_mask = batch.pop(VIDEO_MASK, None)
    prompt_mask = batch.pop(PROMPT_MASK, None)
    prompt_ids_2 = batch.pop(PROMPT_IDS_2, None)
    prompt_mask_2 = batch.pop(PROMPT_MASK_2, None)
    output_tensor_list = model(
        video=video,
        video_mask=video_mask,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        prompt_ids_2=prompt_ids_2,
        prompt_mask_2=prompt_mask_2,
        **batch,
    )
    torch.distributed.barrier()
    loss_dict = unwrap_model(model).compute_loss(*output_tensor_list)
    torch.distributed.barrier()
    return loss_dict, loss_func

# pretrain函数调用datasets_provider, 而pretrain中dataloader_type传external，所以这里返回的iter就是实际用到的iter
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_dataset = build_mm_dataset(args.mm.data.dataset_param)

    enable_encoder_dp = args.mm.model.enable_encoder_dp if hasattr(args.mm.model, "enable_encoder_dp") else False
    # After enabling the encoder dp, data is taken every tp_cp size step, 
    # but the world size of the dataloader is all ranks, 
    # so when calculating the initial global step, 
    # it should be divided by the tp_cp size
    if enable_encoder_dp:
        process_group = torch.distributed.group.WORLD
        encoder_dp_size = torch.distributed.get_world_size(mpu.get_tensor_and_context_parallel_group())
        setattr(args.mm.data.dataloader_param, 'encoder_dp_size', encoder_dp_size)
        print_rank_0(f"use encoder_dp, encoder_dp_size: {args.mm.data.dataloader_param.encoder_dp_size}")
    else:
        process_group = mpu.get_data_parallel_group()

    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=process_group,
    )
    data_iterator, _, _ = build_iterations(train_dl=train_dataloader, iterator_type='single')
    torch.distributed.barrier()
    return data_iterator, None, None


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    data_meta_info_list = [
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/test_data.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/test_data.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data00.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data01.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data02.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data03.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data04.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data05.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data06.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data07.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data08.txt',
        '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data09.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data10.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data11.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data12.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data13.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data14.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data15.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data16.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data17.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data18.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data19.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data20.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data21.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data22.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data23.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data24.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data25.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data26.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data27.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data28.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data29.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data30.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data31.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data32.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data33.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data34.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data35.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data36.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data37.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data38.txt',
        # '/work/share/projects/gyy/mindspeed/Open-Sora-Plan/examples/opensoraplan1.5/data39.txt',
    ]

    pretrain(
        data_meta_info_list,
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False},
    )
    print(f"pretrain done, rank = {torch.distributed.get_rank()}, wait for all processes to exit...")
    torch.distributed.barrier()
    print(f"all processes exit, rank = {torch.distributed.get_rank()}")
    sys.exit(0)
