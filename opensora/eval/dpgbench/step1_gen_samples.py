import os
import torch
import argparse
from PIL import Image
import numpy as np
try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    pass
from opensora.utils.sample_utils import prepare_pipeline, get_args
from opensora.utils.utils import set_seed

def run_model_and_return_samples(
        pipeline, 
        text_prompt, 
        num_frames=1, 
        height=384, 
        width=384, 
        num_sampling_steps=100, 
        guidance_scale=7.0, 
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
        num_samples_per_prompt=num_samples_per_prompt,
        max_sequence_length=max_sequence_length,
        use_linear_quadratic_schedule=True, 
    ).videos
    return videos  # b t h w c, [0, 255]

def concat_image(tensor_images, save_path):
    tensor_images = tensor_images.squeeze(1)

    images_np = tensor_images.detach().cpu().numpy().astype(np.uint8)

    # 创建一个新的空白图像，宽度和高度是单张图像的两倍（假设单张图像尺寸是384x384）
    new_image = Image.new('RGB', (384 * 2, 384 * 2))

    # 将四张图像粘贴到新图像的相应位置
    for index in range(4):
        row = index // 2
        col = index % 2
        img_np = images_np[index]
        img = Image.fromarray(img_np)
        new_image.paste(img, (col * 384, row * 384))

    # 保存拼接后的图像
    new_image.save(save_path)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--version", type=str, default='v1_5', choices=['v1_3', 'v1_5'])
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
    parser.add_argument('--result_path', type=str, default='/storage/hxy/t2i/opensora/Open-Sora-Plan/opensora/eval/dpgbench_test/results')
    parser.add_argument('--prompt_path', type=str, default='/storage/hxy/t2i/opensora/Open-Sora-Plan/opensora/eval/dpgbench_test/ELLA/dpg_bench/prompts')
 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed = 1234
    set_seed(seed, rank=0, device_specific=False)
    device = torch.cuda.current_device()
    pipeline = prepare_pipeline(args, device)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    for filename in os.listdir(args.prompt_path):
        if filename.endswith('.txt'):  # 只处理txt文件
            file_path = os.path.join(args.prompt_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_prompt = file.read()  # 读取文件中的文本
                # all_texts.append(text)  # 将文本添加到列表中
            
            image = run_model_and_return_samples(
            pipeline, 
            text_prompt, 
            height=384, 
            width=384, 
            num_sampling_steps=100, 
            guidance_scale=7.0, 
            num_samples_per_prompt=4, 
            )  # b t h w c, [0, 255]

            img_name = filename.replace('.txt', '.png')

            save_path = os.path.join(args.result_path, img_name)

            concat_image(image, save_path)

    # text_prompt = 'a dog'
    # image = run_model_and_return_samples(
    #     pipeline, 
    #     text_prompt, 
    #     height=384, 
    #     width=384, 
    #     num_sampling_steps=100, 
    #     guidance_scale=7.0, 
    #     num_samples_per_prompt=4, 
    #     )  # b t h w c, [0, 255]
    
    # # import ipdb;ipdb.set_trace()

    # # Image.fromarray(image[0][0].detach().cpu().numpy()).save('test_t3.png')
    # concat_image(image, 'test.png')
