# Copyright 2024 Huawei Technologies Co., Ltd

import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                       torch_dtype=torch.float16).to("npu")
init_image = load_image("cat.png")
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipe(prompt, image=init_image).images[0]
image.save("cat_wizard.png")
