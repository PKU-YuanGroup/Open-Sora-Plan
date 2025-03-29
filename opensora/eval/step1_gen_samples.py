import pandas as pd
import argparse
import torch
import os
from PIL import Image
from tqdm import tqdm
from opensora.utils.utils import set_seed
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    pass
from opensora.utils.sample_utils import init_gpu_env, init_npu_env, prepare_pipeline
from opensora.eval.gen_samples import run_model_and_return_samples
from opensora.eval.general import get_meta
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--version", type=str, default='v1_5', choices=['v1_3', 'v1_5', 't2i'])
    parser.add_argument("--model_type", type=str, default='t2v', choices=['t2v', 'inpaint', 'i2v'])
    parser.add_argument("--ae_dtype", type=str, default='fp16')
    parser.add_argument("--weight_dtype", type=str, default='fp16')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text_encoder_name_1", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--text_encoder_name_3", type=str, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_sampling_steps", type=int, default=24)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument("--sample_method", type=str, default="OpenSoraFlowMatchEuler")
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true') 
    parser.add_argument('--use_pos_neg_prompt', action='store_true') 
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument('--rescale_betas_zero_snr', action='store_true')
    parser.add_argument('--sp', action='store_true')
    parser.add_argument('--allow_tf32', action='store_true')
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if torch_npu is not None:
        npu_config.print_msg(args)
        npu_config.conv_dtype = torch.float16
        init_npu_env(args)
    else:
        args = init_gpu_env(args)

    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(args.seed, rank=args.local_rank, device_specific=True)
    device = torch.cuda.current_device()
    pipeline = prepare_pipeline(args, device)

    meta_info = get_meta(args.prompt_type)
    print(f'origin meta_info ({len(meta_info)})')
    text_and_savepath = [
        [
            meta_info[i]['Prompts'], os.path.join(args.output_dir, f"{meta_info[i]['id']}.jpg")
            ] for i in range(len(meta_info))
        ]

    text_and_savepath_ = [
        [text_prompt, save_path] for text_prompt, save_path in text_and_savepath if not os.path.exists(save_path)
    ]
    print(f'need to process ({len(text_and_savepath_)})')
    if len(text_and_savepath_) == 0:
        import sys;sys.exit(0)
    text_and_savepath = text_and_savepath[args.local_rank::args.world_size]
    os.makedirs(args.output_dir, exist_ok=True)

    cnt = 0
    for text_prompt, save_path in tqdm(text_and_savepath):
        # print(text_prompt, save_path)
        if os.path.exists(save_path):
            continue
        set_seed(args.seed + cnt * 50, rank=args.local_rank, device_specific=True)
        image = run_model_and_return_samples(
            pipeline, 
            text_prompt, 
            height=args.height, 
            width=args.width, 
            num_sampling_steps=args.num_sampling_steps, 
            guidance_scale=args.guidance_scale, 
            guidance_rescale=args.guidance_rescale, 
            num_samples_per_prompt=args.num_samples_per_prompt, 
            use_pos_neg_prompt=args.use_pos_neg_prompt, 
            )  # b t h w c, [0, 255]
        image = image[0][0].detach().cpu().numpy()
        Image.fromarray(image).save(save_path)
        # import ipdb;ipdb.set_trace()
        assert args.num_samples_per_prompt == 1
        cnt += 1


