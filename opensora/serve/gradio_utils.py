import random

import torch


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, 203279)
    return seed

title_markdown = ("""
<div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src='https://www.pnglog.com/AOuPMh.png' style='width: 400px; height: auto; margin-right: 10px;' />
</div>
""")
DESCRIPTION = """
# Open-Sora-Plan v1.1.0
## If Open-Sora-Plan is helpful, please help to ✨ the [Github Repo](https://github.com/PKU-YuanGroup/Open-Sora-Plan) and recommend it to your friends 😊
#### [Open-Sora-Plan v1.1.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan) is a transformer-based text-to-video diffusion system trained on text embeddings from T5.
#### This demo is only trained on 3k hours videos, when creating videos, please be aware that it has the potential to generate harmful videos. For more details read our [report]().
#### Image generation is typically 50 steps, video generation maybe 150 steps will yield good results, but this may take 3-4 minutes.
#### Feel free to enjoy the examples.
#### English prompts ONLY; 提示词仅限英文
####
"""

# <br>
# <div align="center">
#     <div style="display:flex; gap: 0.25rem;" align="center">
#         <a href='https://github.com/PKU-YuanGroup/Open-Sora-Plan'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
#         <a href='https://github.com/PKU-YuanGroup/Open-Sora-Plan/stargazers'><img src='https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social'></a>
#     </div>
# </div>
# """)

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


examples = [
        ["A quiet beach at dawn, the waves gently lapping at the shore and the sky painted in pastel hues.", 50, 5.0],
        ["A quiet beach at dawn, the waves softly lapping at the shore, pink and orange hues painting the sky, offering a moment of solitude and reflection.", 50, 5.0],
        ["The majestic beauty of a waterfall cascading down a cliff into a serene lake.", 50, 5.0],
        ["Sunset over the sea.", 50, 5.0],
        ["a cat wearing sunglasses and working as a lifeguard at pool.", 50, 5.0],
        ["Slow pan upward of blazing oak fire in an indoor fireplace.", 50, 5.0],
        ["Yellow and black tropical fish dart through the sea.", 50, 5.0],
        ["a serene winter scene in a forest. The forest is blanketed in a thick layer of snow, which has settled on the branches of the trees, creating a canopy of white. The trees, a mix of evergreens and deciduous, stand tall and silent, their forms partially obscured by the snow. The ground is a uniform white, with no visible tracks or signs of human activity. The sun is low in the sky, casting a warm glow that contrasts with the cool tones of the snow. The light filters through the trees, creating a soft, diffused illumination that highlights the texture of the snow and the contours of the trees. The overall style of the scene is naturalistic, with a focus on the tranquility and beauty of the winter landscape.", 50, 5.0],
        ["a dynamic interaction between the ocean and a large rock. The rock, with its rough texture and jagged edges, is partially submerged in the water, suggesting it is a natural feature of the coastline. The water around the rock is in motion, with white foam and waves crashing against the rock, indicating the force of the ocean's movement. The background is a vast expanse of the ocean, with small ripples and waves, suggesting a moderate sea state. The overall style of the scene is a realistic depiction of a natural landscape, with a focus on the interplay between the rock and the water.", 50, 5.0],
        ["A serene waterfall cascading down moss-covered rocks, its soothing sound creating a harmonious symphony with nature.", 50, 5.0],
        ["A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures.", 50, 5.0],
        ["The video captures the majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty.", 50, 5.0],
        ["A vibrant scene of a snowy mountain landscape. The sky is filled with a multitude of colorful hot air balloons, each floating at different heights, creating a dynamic and lively atmosphere. The balloons are scattered across the sky, some closer to the viewer, others further away, adding depth to the scene.  Below, the mountainous terrain is blanketed in a thick layer of snow, with a few patches of bare earth visible here and there. The snow-covered mountains provide a stark contrast to the colorful balloons, enhancing the visual appeal of the scene.", 50, 5.0],
        ["A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell, is the main focus of the video, swimming gracefully towards the right side of the frame. The coral reef, teeming with life, is visible in the background, providing a vibrant and colorful backdrop to the turtle's journey. Several small fish, darting around the turtle, add a sense of movement and dynamism to the scene.", 50, 5.0],
        ["A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road.", 50, 5.0],
        ["The dynamic movement of tall, wispy grasses swaying in the wind. The sky above is filled with clouds, creating a dramatic backdrop. The sunlight pierces through the clouds, casting a warm glow on the scene. The grasses are a mix of green and brown, indicating a change in seasons. The overall style of the video is naturalistic, capturing the beauty of the landscape in a realistic manner. The focus is on the grasses and their movement, with the sky serving as a secondary element. The video does not contain any human or animal elements.", 50, 5.0],
]
