import json
import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_path", type=str, required=True, help="The path to the prompt file."
    )
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1_5", choices=["v1_3", "v1_5"])
    parser.add_argument(
        "--model_type", type=str, default="t2v", choices=["t2v", "inpaint", "i2v"]
    )
    parser.add_argument("--ae_dtype", type=str, default="fp16")
    parser.add_argument("--weight_dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--text_encoder_name_1", type=str, default="DeepFloyd/t5-v1_1-xxl"
    )
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--text_encoder_name_3", type=str, default=None)
    parser.add_argument("--sample_method", type=str, default="OpenSoraFlowMatchEuler")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")
    parser.add_argument("--sp", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Initialize the environment
    if torch_npu is not None:
        npu_config.print_msg(args)
        npu_config.conv_dtype = torch.float16
        init_npu_env(args)
    else:
        args = init_gpu_env(args)
    set_seed(args.seed, rank=args.local_rank, device_specific=True)
    device = torch.cuda.current_device()
    print(f"Using device: {device}")
    # Prepare the pipeline
    pipeline = prepare_pipeline(args, device)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the evaluation prompts
    with open(args.prompt_path, "r") as f:
        metadatas = [json.loads(line) for line in f]

    inference_list = []
    
    for index, metadata in enumerate(metadatas):
        outpath = os.path.join(args.output_dir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)
        all_samples = list()
        
        for idx, n in enumerate(range(args.n_samples)):
            inference_list.append([prompt, sample_path, idx])
            
    inference_list = inference_list[args.local_rank::args.world_size]
    for prompt, sample_path, sample_count in tqdm(inference_list):
        image = run_model_and_return_samples(
            pipeline,
            prompt,
            height=args.height,
            width=args.width,
            num_sampling_steps=args.num_sampling_steps,
            guidance_scale=args.cfg,
            num_samples_per_prompt=1,
            use_pos_neg_prompt=False
        )
        image = image[0][0].detach().cpu().numpy()
        # Save image
        Image.fromarray(image).save(
            os.path.join(sample_path, f"{sample_count:05}.png")
        )
