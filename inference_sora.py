import os

import torch
import mindspeed.megatron_adaptor
import torch.distributed as dist
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.inference.pipeline import OpenSoraPlanPipeline
from mindspeed_mm.inference.pipeline.utils.sora_utils import (
    save_image_or_videos,
    save_image_or_video_grid,
    load_prompts,
    load_images,
    load_conditional_pixel_values
)
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm import Tokenizer
from mindspeed_mm.utils.utils import get_dtype, get_device, is_npu_available
from mindspeed_mm.data.data_utils.utils import DecordDecoder
import json
import random
import shutil
import numpy as np
from tqdm import tqdm 

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False

def prepare_pipeline(args, dtype, ae_dtype, device):
    ori_args = get_args()
    vae = AEModel(args.ae).to(device=device).eval()
    text_encoder = TextEncoder(args.text_encoder, dtype=dtype).get_model().to(device=device).eval()
    text_encoder_2 = TextEncoder(args.text_encoder_2, dtype=dtype).get_model().to(device=device).eval() if args.text_encoder_2 is not None else None
    predict_model = PredictModel(args.predictor).get_model()
    if ori_args.load is not None:
        load_checkpoint([predict_model], None, None, strict=False)
    predict_model = predict_model.to(device=device, dtype=dtype).eval()
    scheduler = DiffusionModel(args.diffusion).get_model()
    tokenizer = Tokenizer(args.tokenizer).get_tokenizer()
    tokenizer_2 = Tokenizer(args.tokenizer_2).get_tokenizer() if args.tokenizer_2 is not None else None
    sora_pipeline_class = OpenSoraPlanPipeline
    sora_pipeline = sora_pipeline_class(vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2, scheduler=scheduler, predict_model=predict_model, config=args.pipeline_config)
    return sora_pipeline


def main():
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)
    args = args.mm.model
    # prepare arguments
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.weight_dtype)
    ae_dtype = get_dtype(args.ae.dtype)
    device = get_device(args.device)

    # json_path = "/work/share1/video_final/tiyu_20241006_final_83899.json"
    # root_dir = "/work/share/dataset/xigua_video"
    # save_dir = "/work/share/projects/gyy/mindspeed/Open-Sora-Plan/test_dataset_to_frame"

    # os.makedirs(save_dir, exist_ok=True)

    # with open(json_path, 'r') as f:
    #     data = json.load(f)

    # test_sample_nums = 100
    # data = random.sample(data, test_sample_nums * 10)

    # for idx in tqdm(range(test_sample_nums), desc='test'):

    #     save_sub_dir = os.path.join(save_dir, f'{idx:06d}')
    #     os.makedirs(save_sub_dir, exist_ok=True)
    #     sample = random.sample(data, 1)[0]
    #     video = os.path.join(root_dir, sample['path'])
    #     while not os.path.exists(video):
    #         sample = random.sample(data, 1)[0]
    #         video = os.path.join(root_dir, sample['path'])

    #     vframes = DecordDecoder(video)
    #     start_frame_idx, end_frame_idx = sample.get('cut', None)
    #     s_x, e_x, s_y, e_y = sample.get('crop', [None, None, None, None])
    #     frame_indice = np.arange(start_frame_idx, end_frame_idx)
    #     clip = vframes.get_batch(frame_indice)
    #     if clip is not None:
    #         if s_y is not None:
    #             clip = clip[:, s_y: e_y, s_x: e_x, :]
    #     clip = torch.unsqueeze(clip, 0)
    #     num_frames = clip.shape[1]
    #     rand_frame = random.randint(0, num_frames)
    #     clip = clip[:, rand_frame: rand_frame + 1, :, :, :]
    #     save_image_or_videos(clip, save_sub_dir, idx)
    #     text = sample["cap"]
    #     if not isinstance(text, list):
    #         text = [text]
    #     with open(os.path.join(save_sub_dir, 'caption.txt'), 'w') as f:
    #         for t in text:
    #             f.write(f'{t}\n')

    prompts = load_prompts(args.prompt)
    images = load_images(args.image) if hasattr(args, "image") else None
    conditional_pixel_values_path = load_conditional_pixel_values(args.conditional_pixel_values_path) if hasattr(args, "conditional_pixel_values_path") else None
    mask_type = args.mask_type if hasattr(args, "mask_type") else None
    crop_for_hw = args.crop_for_hw if hasattr(args, "crop_for_hw") else None
    max_hxw = args.max_hxw if hasattr(args, "max_hxw") else None
    num_samples_per_prompt = args.num_samples_per_prompt if hasattr(args, "num_samples_per_prompt") else 1

    if images is not None and len(prompts) != len(images):
        raise AssertionError(f'The number of images {len(images)} and the numbers of prompts {len(prompts)} do not match')

    if len(prompts) % args.micro_batch_size != 0:
        raise AssertionError(f'The number of  prompts {len(prompts)} is not divisible by the batch size {args.micro_batch_size}')

    save_fps = args.fps // args.frame_interval

    # prepare pipeline
    sora_pipeline = prepare_pipeline(args, dtype, ae_dtype, device)

    # == Iter over all samples ==
    video_grids = []
    start_idx = 0
    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()
    for i in range(rank * args.micro_batch_size, len(prompts), args.micro_batch_size * world_size):
        # == prepare batch prompts ==
        batch_prompts = prompts[i: i + args.micro_batch_size]
        kwargs = {}
        if conditional_pixel_values_path:
            batch_pixel_values_path = conditional_pixel_values_path[i: i + args.micro_batch_size]
            kwargs.update({"conditional_pixel_values_path": batch_pixel_values_path,
                           "mask_type": mask_type,
                           "crop_for_hw": crop_for_hw,
                           "max_hxw": max_hxw})

        if images is not None:
            batch_images = images[i: i + args.micro_batch_size]
        else:
            batch_images = None

        videos = sora_pipeline(prompt=batch_prompts,
                               image=batch_images,
                               fps=save_fps,
                               max_sequence_length=args.model_max_length,
                               use_prompt_preprocess=True,
                               num_samples_per_prompt=num_samples_per_prompt,
                               device=device,
                               dtype=dtype,
                               **kwargs
                               )
        if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            save_image_or_videos(videos, args.save_path, start_idx + rank, save_fps)
            start_idx += len(batch_prompts) * world_size
            video_grids.append(videos)
    if len(video_grids) > 0:
        video_grids = torch.cat(video_grids, dim=0).to(device)

    if len(prompts) < args.micro_batch_size * world_size:
        active_ranks = range(len(prompts) // args.micro_batch_size)
    else:
        active_ranks = range(world_size)
    active_ranks = [x * mpu.get_tensor_model_parallel_world_size() * mpu.get_context_parallel_world_size() for x in active_ranks]

    dist.barrier()
    gathered_videos = []
    rank = dist.get_rank()
    if rank == 0:
        for r in active_ranks:
            if r != 0:  # main process does not need to receive from itself
                # receive tensor shape
                shape_tensor = torch.empty(5, dtype=torch.int, device=device)
                dist.recv(shape_tensor, src=r)
                shape_videos = shape_tensor.tolist()

                # create receiving buffer based on received shape
                received_videos = torch.empty(shape_videos, dtype=video_grids.dtype, device=device)
                dist.recv(received_videos, src=r)
                gathered_videos.append(received_videos.cpu())
            else:
                gathered_videos.append(video_grids.cpu())
    elif rank in active_ranks:
        # send tensor shape first
        shape_tensor = torch.tensor(video_grids.shape, dtype=torch.int, device=device)
        dist.send(shape_tensor, dst=0)

        # send the tensor
        dist.send(video_grids, dst=0)
    dist.barrier()
    if rank == 0:
        video_grids = torch.cat(gathered_videos, dim=0)
        save_image_or_video_grid(video_grids, args.save_path, save_fps)
        print("Inference finished.")
        print("Saved %s samples to %s" % (video_grids.shape[0], args.save_path))


if __name__ == "__main__":
    main()
