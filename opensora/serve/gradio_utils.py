import random

import imageio
import uuid
import torch

import numpy as np


POS_PROMPT = """
    high quality, high aesthetic, {}
    """

NEG_PROMPT = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

NUM_IMAGES_PER_PROMPT = 1
MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


LOGO = """
    <center><img src='https://s21.ax1x.com/2024/07/14/pk5pLBF.jpg' alt='Open-Sora Plan logo' style="width:220px; margin-bottom:1px"></center>
"""
TITLE = """
    <div style="text-align: center; font-size: 45px; font-weight: bold; margin-bottom: 5px;">
        Open-Sora Plan🤗
    </div>
"""
DESCRIPTION = """
    <div style="text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
        Support Chinese and English; 支持中英双语
    </div>
    <div style="text-align: center; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
        Welcome to Star🌟 our <a href='https://github.com/PKU-YuanGroup/Open-Sora-Plan' target='_blank'><b>GitHub</b></a>
    </div>
"""

t2v_prompt_examples = [
    "An animated scene features a close-up of a short, fluffy monster kneeling beside a melting red candle. The 3D, realistic art style focuses on the interplay of lighting and texture, casting intriguing shadows across the scene. The monster gazes at the flame with wide, curious eyes, its fur gently ruffling in the warm, flickering glow. The camera slowly zooms in, capturing the intricate details of the monster's fur and the delicate, molten wax droplets. The atmosphere is filled with a sense of wonder and curiosity, as the monster tentatively reaches out a paw, as if to touch the flame, while the candlelight dances and flickers around it.", 
    "动画场景特写中，一个矮小、毛茸茸的怪物跪在一根融化的红蜡烛旁。三维写实的艺术风格注重光照和纹理的相互作用，在整个场景中投射出引人入胜的阴影。怪物睁着好奇的大眼睛注视着火焰，它的皮毛在温暖闪烁的光芒中轻轻拂动。镜头慢慢拉近，捕捉到怪物皮毛的复杂细节和精致的熔蜡液滴。怪物试探性地伸出一只爪子，似乎想要触碰火焰，而烛光则在它周围闪烁舞动，气氛充满了惊奇和好奇。", 
    "A close-up shot captures a Victoria crowned pigeon, its striking blue plumage and vibrant red chest standing out prominently. The bird's delicate, lacy crest and striking red eye add to its regal appearance. The pigeon's head is tilted slightly to the side, giving it a majestic look. The background is blurred, drawing attention to the bird's striking features. Soft light bathes the scene, casting gentle shadows that enhance the texture of its feathers. The pigeon flutters its wings slightly, and its beak tilts upwards, as if curiously observing the surroundings, creating a dynamic and captivating atmosphere.", 
    "特写镜头捕捉到一只维多利亚皇冠鸽，其醒目的蓝色羽毛和鲜艳的红色胸部格外显眼。这只鸽子精致的花边鸽冠和醒目的红眼更增添了它的威严。鸽子的头部略微偏向一侧，给人一种威严的感觉。背景被模糊处理，使人们的注意力集中在鸽子引人注目的特征上。柔和的光线洒在画面上，投下柔和的阴影，增强了鸽子羽毛的质感。鸽子微微扇动翅膀，嘴角向上翘起，似乎在好奇地观察周围的环境，营造出一种动感迷人的氛围。"
]




style_list = [
    {
        "name": "(Default)",
        "prompt": "(masterpiece), (best quality), (ultra-detailed), (unwatermarked), {prompt}",
        "negative_prompt": NEG_PROMPT,
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured. ",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo, a close-up of  {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly. ",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast. ",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style. ",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly. ",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic. ",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white. ",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured. ",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting. ",
    },
]
