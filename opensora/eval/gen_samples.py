import os
import torch
import argparse
from PIL import Image
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    pass
from opensora.utils.sample_utils import prepare_pipeline
from opensora.utils.utils import set_seed

def run_model_and_return_samples(
        pipeline, 
        text_prompt, 
        num_frames=1, 
        height=384, 
        width=384, 
        num_sampling_steps=100, 
        guidance_scale=7.0, 
        guidance_rescale=0.7, 
        num_samples_per_prompt=1, 
        max_sequence_length=512, 
        use_pos_neg_prompt=True, 
        ):

    if use_pos_neg_prompt:
        positive_prompt = """
        high quality, high aesthetic, {}
        """
        negative_prompt = """
        nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
        low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
        """
    else:
        positive_prompt = "{}"
        negative_prompt = ""
    
    videos = pipeline(
        positive_prompt.format(text_prompt), 
        negative_prompt=negative_prompt, 
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_sampling_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        num_samples_per_prompt=num_samples_per_prompt,
        max_sequence_length=max_sequence_length,
        use_linear_quadratic_schedule=True, 
    ).videos
    return videos  # b t h w c, [0, 255]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default='v1_5', choices=['v1_3', 'v1_5', 't2i'])
    parser.add_argument("--model_type", type=str, default='t2v', choices=['t2v', 'inpaint', 'i2v'])
    parser.add_argument("--ae_dtype", type=str, default='fp16')
    parser.add_argument("--weight_dtype", type=str, default='fp16')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name_1", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--text_encoder_name_3", type=str, default=None)
    parser.add_argument("--sample_method", type=str, default="OpenSoraFlowMatchEuler")
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--save_memory', action='store_true') 
    parser.add_argument("--prediction_type", type=str, default='epsilon', help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument('--rescale_betas_zero_snr', action='store_true')
 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed = 1234
    set_seed(seed, rank=0, device_specific=False)
    device = torch.cuda.current_device()
    pipeline = prepare_pipeline(args, device)

    text_prompt = 'a dog'
    image = run_model_and_return_samples(
        pipeline, 
        text_prompt, 
        height=384, 
        width=384, 
        num_sampling_steps=100, 
        guidance_scale=7.0, 
        num_samples_per_prompt=1, 
        )  # b t h w c, [0, 255]
    Image.fromarray(image[0][0].detach().cpu().numpy()).save('test_t3_doubleffn.png')
