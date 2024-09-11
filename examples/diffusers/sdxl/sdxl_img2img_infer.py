# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2023 The HuggingFace Team. All rights reserved.

import datetime
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np
import PIL.Image as Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
VUE_NAME = "madebyollin/sdxl-vae-fp16-fix"

time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
print("start time: " + time)

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("npu")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
         "diffusers/controlnet-depth-sdxl-1.0-small",
         variant="fp16",
         use_safetensors=True,
         torch_dtype=torch.float16,
).to("npu")
vae = AutoencoderKL.from_pretrained(VUE_NAME, torch_dtype=torch.float16).to("npu")
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
         MODEL_NAME,
         controlnet=controlnet,
         vae=vae,
         variant="fp16",
         use_safetensors=True,
         torch_dtype=torch.float16,
).to("npu")
pipe.enable_model_cpu_offload()


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("npu")
    with torch.no_grad(), torch.autocast("npu"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


prompt = "A robot, 4k photo"
pre_image = load_image("cat.png").resize((1024, 1024))
controlnet_conditioning_scale = 0.5  # recommended for good generalization
depth_image = get_depth_map(pre_image)

seed_list = [8, 23, 42, 1334]
for i in seed_list:  
    generator = torch.Generator(device="cpu").manual_seed(i)
    images = pipe(
        prompt,
        image=pre_image,
        control_image=depth_image,
        strength=0.99,
        num_inference_steps=50,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    ).images
    images[0].save(f"robot_cat-NPU-{i}.png")

time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
print("end time: " + time)

