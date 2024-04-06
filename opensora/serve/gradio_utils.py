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
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">

   <a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="https://www.pnglog.com/AOuPMh.png" alt="Open-Sora-Plan🚀" style="max-width: 200px; height: auto;">
  </a>
  <div>
    <h1 >Open-Sora-Plan v1.0.0</h1>
    <h3 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h3>
  </div>
</div>
<br>
<div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/PKU-YuanGroup/Open-Sora-Plan'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        <a href='https://github.com/PKU-YuanGroup/Open-Sora-Plan/stargazers'><img src='https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social'></a>
    </div>
</div>
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


examples = [
    [
        "A soaring drone footage captures the majestic beauty of a coastal cliff, "
        "its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. "
        "Seabirds can be seen taking flight around the cliff's precipices. "
        "As the drone slowly moves from different angles, "
        "the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. "
        "The water gently laps at the rock base and the greenery that clings to the top of the cliff, "
        "and the scene gives a sense of peaceful isolation at the fringes of the ocean. "
        "The video captures the essence of pristine natural beauty untouched by human structures.",
        50, 7.5,
    ],
    [
        "The video captures the majestic beauty of a waterfall cascading down a cliff into a serene lake. "
        "The waterfall, with its powerful flow, is the central focus of the video. "
        "The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. "
        "The camera angle provides a bird's eye view of the waterfall, "
        "allowing viewers to appreciate the full height and grandeur of the waterfall. "
        "The video is a stunning representation of nature's power and beauty.",
        50, 7.5,
    ],
    [
        "A vibrant scene of a snowy mountain landscape. "
        "The sky is filled with a multitude of colorful hot air balloons, "
        "each floating at different heights, creating a dynamic and lively atmosphere. "
        "The balloons are scattered across the sky, some closer to the viewer, "
        "others further away, adding depth to the scene.  "
        "Below, the mountainous terrain is blanketed in a thick layer of snow, "
        "with a few patches of bare earth visible here and there. "
        "The snow-covered mountains provide a stark contrast to the colorful balloons, "
        "enhancing the visual appeal of the scene.  In the foreground, "
        "a few cars can be seen driving along a winding road that cuts through the mountains. "
        "The cars are small compared to the vastness of the landscape, emphasizing the grandeur of the surroundings. "
        "The overall style of the video is a mix of adventure and tranquility, "
        "with the hot air balloons adding a touch of whimsy to the otherwise serene mountain landscape. "
        "The video is likely shot during the day, as the lighting is bright and even, "
        "casting soft shadows on the snow-covered mountains.",
        50, 7.5,
    ]
]