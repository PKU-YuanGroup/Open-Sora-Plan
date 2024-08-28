# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2023 The HuggingFace Team. All rights reserved.

import random
import os
from diffusers import DiffusionPipeline
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np


output_path = "./lora_nodp_NPU3"
os.makedirs(output_path, exist_ok=True)



model_path = "/diffusion/sdxl/model/"
pipe = DiffusionPipeline.from_pretrained("/diffusion/sdxl/pretrained/", torch_dtype=torch.float32)
pipe.to("npu")


prompts = dict()
prompts = {
    "masterpiece, best quality, Cute dragon creature, pokemon style, night, moonlight, dim lighting" : "deformed, disfigured, underexposed, overexposed, rugged, (low quality), (normal quality),",
    "masterpiece, best quality, Pikachu walking in beijing city, pokemon style, night, moonlight, dim lighting" : "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),",
    "masterpiece, best quality, red panda , pokemon style, evening light, sunset, rim lighting" : "deformed, disfigured, underexposed, overexposed, (low quality), (normal quality),",
    "masterpiece, best quality, Photo of (Lion:1.2) on a couch, flower in vase, dof, film grain, crystal clear, pokemon style, dark studio" : "deformed, disfigured, underexposed, overexposed,",
    "masterpiece, best quality, siberian cat pokemon on river, pokemon style, evening light, sunset, rim lighting, depth of field" : "deformed, disfigured, underexposed, overexposed,",
    "masterpiece, best quality, pig, Exquisite City, (sky:1.3), (Miniature tree:1.3), Miniature object, many flowers, glowing mushrooms, (creek:1.3), lots of fruits, cute colorful animal protagonist, Firefly, meteor, Colorful cloud, pokemon style, Complicated background, rainbow," : "Void background,black background,deformed, disfigured, underexposed, overexposed,",
    "masterpiece, best quality, (pokemon), a cute pikachu, girl with glasses, (masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2)," : "(low quality), (normal quality), (monochrome), lowres, extra fingers, fewer fingers, (watermark),",
    "masterpiece, best quality, sugimori ken \(style\), (pokemon \(creature\)), pokemon electric type, grey and yellow skin, mechanical arms, cyberpunk city background, night, neon light" : "(worst quality, low quality:1.4), watermark, signature, deformed, disfigured, underexposed, overexposed,"
}
#设置随机数种子
seed_list = [8, 23, 42, 1334]

#输出图片
for prompt_key, negative_prompt_key in prompts.items():
    for i in seed_list:    
        generator = torch.Generator(device="cpu").manual_seed(i)
        image = pipe(prompt=prompt_key, negative_prompt=negative_prompt_key, generator=generator).images
        image[0].save(f"{output_path}/{prompt_key[26:40]}-{i}.png")