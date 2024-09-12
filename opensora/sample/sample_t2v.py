import os
import torch
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    pass
from opensora.utils.sample_utils import (
    init_gpu_env, init_npu_env, prepare_pipeline, get_args, 
    run_model_and_save_samples, run_model_and_save_samples_npu, 
    prepare_caption_refiner
)

if __name__ == "__main__":
    args = get_args()
    if torch_npu is not None:
        npu_config.print_msg(args)
        npu_config.conv_dtype = torch.bfloat16
        init_npu_env(args)
    else:
        args = init_gpu_env(args)

    device = torch.cuda.current_device()
    pipeline = prepare_pipeline(args, device)
    if args.caption_refiner is not None:
        caption_refiner_model, caption_refiner_tokenizer = prepare_caption_refiner(args, device)
    else:
        caption_refiner_model, caption_refiner_tokenizer = None, None

    if npu_config is not None and npu_config.on_npu and npu_config.profiling:
        run_model_and_save_samples_npu(args, pipeline, caption_refiner_model, caption_refiner_tokenizer)
    else:
        run_model_and_save_samples(args, pipeline, caption_refiner_model, caption_refiner_tokenizer)
