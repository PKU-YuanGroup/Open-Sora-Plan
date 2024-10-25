import os

import torch
import mindspeed.megatron_adaptor
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.core import mpu

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.tasks.inference.pipeline import SoraPipeline_dict
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import save_videos, save_one_video, load_prompts, load_conditional_pixel_values_path
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm import Tokenizer
from mindspeed_mm.utils.utils import get_dtype, get_device, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_pipeline(args, device):
    vae = AEModel(args.ae).get_model().to(device, args.ae.dtype).eval()
    text_encoder = TextEncoder(args.text_encoder).get_model().to(device).eval()
    predict_model = PredictModel(args.predictor).get_model().to(device, args.predictor.dtype).eval()
    scheduler = DiffusionModel(args.diffusion).get_model()
    tokenizer = Tokenizer(args.tokenizer).get_tokenizer()
    if not hasattr(vae, 'dtype'):
        vae.dtype = args.ae.dtype
    tokenizer.model_max_length = args.model_max_length
    sora_pipeline_class = SoraPipeline_dict[args.pipeline_class]
    sora_pipeline = sora_pipeline_class(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                                        predict_model=predict_model, config=args.pipeline_config)
    return sora_pipeline


def main():
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)
    args = args.mm.model
    # prepare arguments
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.dtype)
    device = get_device(args.device)

    prompts = load_prompts(args.prompt)
    if "Inpaint" in args.pipeline_class:
        conditional_pixel_values_path = load_conditional_pixel_values_path(args.conditional_pixel_values_path)
    start_idx = 0
    max_sequence_length = args.model_max_length
    save_fps = args.fps // args.frame_interval
    os.makedirs(args.save_path, exist_ok=True)

    # prepare pipeline
    sora_pipeline = prepare_pipeline(args, device)
    torch.manual_seed(mpu.get_context_parallel_rank())

    # == Iter over all samples ==
    video_grids = []
    if "Inpaint" in args.pipeline_class:
        print("Inpainting mode")
        for i in range(0, len(prompts)):
            prompt = prompts[i]
            condition = conditional_pixel_values_path[i]
            videos = sora_pipeline(prompt=prompt, conditional_pixel_values_path=condition, fps=save_fps, device=device,
                                   dtype=dtype, max_sequence_length=max_sequence_length)
            save_one_video(videos[0], args.save_path, save_fps, start_idx)
            video_grids.append(videos)
            start_idx += 1
    else:
        print("T2V mode")
        for i in range(0, len(prompts)):
            # == prepare batch prompts ==
            prompt = prompts[i]
            videos = sora_pipeline(prompt=prompt, fps=save_fps, device=device, dtype=dtype,
                                max_sequence_length=max_sequence_length)
            save_one_video(videos[0], args.save_path, save_fps, start_idx)
            video_grids.append(videos)
            start_idx += 1
    video_grids = torch.cat(video_grids, dim=0)
    if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        save_videos(video_grids, args.save_path, save_fps, value_range=(-1, 1), normalize=True)
        print("Inference finished.")
        print(f"Saved {start_idx} samples to {args.save_path}")


if __name__ == "__main__":
    main()
