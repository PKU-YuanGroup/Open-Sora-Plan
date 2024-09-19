import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import KolorsPipeline


pipe = KolorsPipeline.from_pretrained(
    "Kwai-Kolors/Kolors-diffusers",
    torch_dtype=torch.float16,
    variant="fp16"
).to("npu")
prompt = '一对年轻的中国情侣，皮肤白皙，穿着时尚的运动装，背景是现代的北京城市天际线。面部细节，清晰的毛孔，使用最新款的相机拍摄，特写镜头，超高画质，8K，视觉盛宴'
image = pipe(
    prompt=prompt,
    negative_prompt="",
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator(pipe.device).manual_seed(66),
).images[0]
image.save('infer_kolors_fp16.png')