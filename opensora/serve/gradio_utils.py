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
    "动画场景特写中，一个矮小、毛茸茸的怪物跪在一根融化的红蜡烛旁。三维写实的艺术风格注重光照和纹理的相互作用，在整个场景中投射出引人入胜的阴影。怪物睁着好奇的大眼睛注视着火焰，它的皮毛在温暖闪烁的光芒中轻轻拂动。镜头慢慢拉近，捕捉到怪物皮毛的复杂细节和精致的熔蜡液滴。怪物试探性地伸出一只爪子，似乎想要触碰火焰，而烛光则在它周围闪烁舞动，气氛充满了惊奇和好奇。", 
    "An animated scene features a close-up of a short, fluffy monster kneeling beside a melting red candle. The 3D, realistic art style focuses on the interplay of lighting and texture, casting intriguing shadows across the scene. The monster gazes at the flame with wide, curious eyes, its fur gently ruffling in the warm, flickering glow. The camera slowly zooms in, capturing the intricate details of the monster's fur and the delicate, molten wax droplets. The atmosphere is filled with a sense of wonder and curiosity, as the monster tentatively reaches out a paw, as if to touch the flame, while the candlelight dances and flickers around it.", 
    "特写镜头捕捉到一只维多利亚皇冠鸽，其醒目的蓝色羽毛和鲜艳的红色胸部格外显眼。这只鸽子精致的花边鸽冠和醒目的红眼更增添了它的威严。鸽子的头部略微偏向一侧，给人一种威严的感觉。背景被模糊处理，使人们的注意力集中在鸽子引人注目的特征上。柔和的光线洒在画面上，投下柔和的阴影，增强了鸽子羽毛的质感。鸽子微微扇动翅膀，嘴角向上翘起，似乎在好奇地观察周围的环境，营造出一种动感迷人的氛围。", 
    "A close-up shot captures a Victoria crowned pigeon, its striking blue plumage and vibrant red chest standing out prominently. The bird's delicate, lacy crest and striking red eye add to its regal appearance. The pigeon's head is tilted slightly to the side, giving it a majestic look. The background is blurred, drawing attention to the bird's striking features. Soft light bathes the scene, casting gentle shadows that enhance the texture of its feathers. The pigeon flutters its wings slightly, and its beak tilts upwards, as if curiously observing the surroundings, creating a dynamic and captivating atmosphere.", 
    "一架无人机捕捉到了大苏尔加雷角海滩上海浪拍打着崎岖悬崖的壮丽景色。湛蓝的海水拍打出白色的浪花，夕阳的金光照亮了岩石海岸，投下长长的阴影，营造出温暖宁静的氛围。远处矗立着一座小岛，岛上有一座灯塔，更增添了画面的魅力。海鸥在头顶上滑翔，海风吹过附近的植被，沙沙作响，给宁静的海岸景观带来了勃勃生机。", 
    "A drone captures a breathtaking view of waves crashing against the rugged cliffs along Big Sur's Garay Point Beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore, casting long shadows and creating a warm, serene atmosphere. A small island with a lighthouse stands in the distance, adding to the scene's charm. Seagulls glide overhead as the ocean breeze rustles through the nearby vegetation, bringing life to the tranquil coastal landscape.", 
    "一个二十岁出头的年轻人，头发蓬松，鼻梁上架着一副眼镜，安详地坐在高高飘扬的蓬松白云上。他全神贯注地读着一本书，偶尔抬起头看一眼周围翱翔的鸟儿。阳光透过飘渺的云层，在这幅画面上洒下柔和的金色光芒，并在他的脸上投下俏皮的影子。当他翻开书页时，一阵微风吹过，书页沙沙作响，他微笑着，感受着失重和自由的快感。", 
    "A young man in his early twenties, with tousled hair and a pair of glasses perched on the end of his nose, sits serenely on a fluffy, white cloud floating high in the sky. He is engrossed in a book, occasionally glancing up to watch the birds soar around him. The sunlight filters through the wispy clouds, casting a soft, golden glow over the scene and creating playful shadows that dance on his face. As he turns a page, a gentle breeze rustles the pages, and he smiles, feeling the thrill of weightlessness and freedom.", 
    "三维动画描绘了一只圆滚滚、毛茸茸的小动物，它有一双富于表情的大眼睛，正在探索一片生机勃勃的魔法森林。这个异想天开的生物是兔子和松鼠的混合体，长着柔软的蓝色皮毛和浓密的条纹尾巴。它沿着波光粼粼的溪流蹦蹦跳跳，眼睛睁得大大的，充满了好奇。森林里充满了神奇的元素：会发光和变色的花朵、长着紫色和银色树叶的树木，还有像萤火虫一样的小浮光。它跳着跳着，停了下来，与一群围着蘑菇圈跳舞的小精灵嬉戏互动。然后，它抬头敬畏地看着一棵发光的大树，这棵树似乎是森林的核心。摄像机平稳地摇镜头，捕捉到这只小动物好奇地伸手触摸一朵发光的花朵，花朵随之变色。整个场景沐浴在柔和、空灵的光线中，背景中的阴影轻轻舞动，营造出一种令人陶醉和惊奇的氛围。小动物的嬉戏打闹和神奇的氛围让森林变得生机勃勃，仿佛每一刻都是一次发现和喜悦。", 
    "A 3D animation depicts a small, round, fluffy creature with big, expressive eyes exploring a vibrant, enchanted forest. This whimsical creature, a blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. As the creature hops, it pauses to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. It then looks up in awe at a large, glowing tree that seems to be the heart of the forest. The camera pans smoothly to capture the creature's curiosity as it reaches out to touch a glowing flower, causing it to change colors. The scene is bathed in a soft, ethereal light, with shadows dancing gently in the background, creating an atmosphere of enchantment and wonder. The creature's playful antics and the magical ambiance make the forest come alive, as if every moment is a discovery and a delight.", 
    "一架无人机优雅地环绕着阿马尔菲海岸崎岖不平的山顶上一座历史悠久的教堂，拍摄其宏伟的建筑细节以及层层叠叠的小径和天井。下方，海浪拍打着岩石，地平线延伸至意大利的沿海水域和丘陵地貌。远处的身影在天井中漫步，欣赏着壮丽的海景，营造出一幅动感十足的画面。午后和煦的阳光让整个场景沐浴在神奇而浪漫的光影中，投下长长的阴影，为迷人的景色增添了深度。镜头不时拉近以突出教堂错综复杂的细节，然后拉远以展示广阔的海岸线，营造出引人入胜的视觉叙事效果。", 
    "A drone camera gracefully circles a historic church perched on a rugged outcropping along the Amalfi Coast, capturing its magnificent architectural details and tiered pathways and patios. Below, waves crash against the rocks, while the horizon stretches out over the coastal waters and hilly landscapes of Italy. Distant figures stroll and enjoy the breathtaking ocean views from the patios, creating a dynamic scene. The warm glow of the afternoon sun bathes the scene in a magical and romantic light, casting long shadows and adding depth to the stunning vista. The camera occasionally zooms in to highlight the intricate details of the church, then pans out to showcase the expansive coastline, creating a captivating visual narrative.", 
    "一个特写镜头捕捉到一位 60 多岁、留着胡子的白发老人，他坐在巴黎的一家咖啡馆里陷入沉思，思考着宇宙的历史。他的眼睛紧紧盯着屏幕外走动的人们，而自己却一动不动。他身着羊毛大衣、纽扣衬衫、棕色贝雷帽，戴着一副眼镜，散发着教授的风范。他偶尔瞥一眼四周，目光停留在背景中熙熙攘攘的巴黎街道和城市景观上。场景沐浴在金色的光线中，让人联想到 35 毫米电影胶片。当他微微前倾时，眼睛睁大，露出顿悟的瞬间，并微微闭口微笑，暗示他已经找到了生命奥秘的答案。景深营造出光影交错的动态效果，烘托出智慧沉思的氛围。", 
    "An extreme close-up captures a gray-haired man with a beard in his 60s, deep in thought as he sits at a Parisian cafe, contemplating the history of the universe. His eyes focus intently on people walking offscreen, while he remains mostly motionless. Dressed in a wool coat, a button-down shirt, a brown beret, and glasses, he exudes a professorial demeanor. The man occasionally glances around, his gaze lingering on the bustling Parisian streets and cityscape in the background. The scene is bathed in golden light, reminiscent of a cinematic 35mm film. As he leans forward slightly, his eyes widen in a moment of epiphany, and he offers a subtle, closed-mouth smile, suggesting he has found the answer to the mystery of life. The depth of field creates a dynamic interplay of light and shadow, enhancing the atmosphere of intellectual contemplation.", 
    "一只欢快的水獭穿着明黄色的救生衣，自信地在冲浪板上保持平衡，在郁郁葱葱的热带岛屿附近波光粼粼的绿松石水域中滑行。该场景采用三维数字艺术风格渲染，阳光在水面上投下俏皮的阴影。水獭不时将爪子伸入水中，溅起的水珠捕捉到光线，为宁静的氛围增添了动感和刺激。", 
    "A cheerful otter confidently balances on a surfboard, donning a bright yellow lifejacket, as it glides through the shimmering turquoise waters near lush tropical islands. The scene is rendered in a 3D digital art style, with the sunlight casting playful shadows on the water's surface. The otter occasionally dips its paws into the water, sending up sprays of droplets that catch the light, adding a sense of motion and excitement to the tranquil atmosphere.", 
    "在这幅迷人的特写镜头中，一只变色龙展示了它非凡的变色能力，在柔和的散射光中，它鲜艳的色调微妙地变换着。模糊的背景凸显了变色龙醒目的外表，而光影的交错则突出了变色龙皮肤的复杂细节。", 
    "In this captivating close-up shot, a chameleon displays its remarkable color-changing abilities, its vibrant hues shifting subtly in the soft, diffused light. The blurred background highlights the animal's striking appearance, while the interplay of light and shadow accentuates the intricate details of its skin.", 
    "圣托里尼在蓝色时刻的壮丽鸟瞰图捕捉到了白色基克拉迪建筑与蓝色圆顶的迷人建筑，在黄昏的天空中投射出长长的阴影。火山口的景色令人惊叹，光与影的交织营造出宁静的氛围。当太阳落到地平线以下时，夕阳的余晖将整个场景笼罩在温暖的金色中，海鸥在空中优雅地翱翔，几艘帆船在下方的火山口悠闲地漂流。", 
    "A breathtaking aerial view of Santorini during the blue hour captures the stunning architecture of white Cycladic buildings with blue domes, casting long shadows against the twilight sky. The caldera views are awe-inspiring, with the interplay of light and shadow creating a serene atmosphere. As the sun dips below the horizon, the gentle glow of the setting sun bathes the scene in a warm, golden hue, while seagulls soar gracefully through the air and a few sailboats drift lazily in the caldera below.", 
    "一群羊驼在鲜艳的涂鸦墙前自信地摆着姿势，每只羊驼都穿着五颜六色的羊毛针织衫，戴着时尚的太阳镜。在正午明媚的阳光下，它们嬉戏互动，有的好奇地东张西望，有的则亲昵地偎依在一起。光与影的鲜明对比增强了这一场景的动感活力，营造出一种融合了都市前卫与奇异魅力的氛围。", 
    "A group of alpacas, each donning colorful knit wool sweaters and stylish sunglasses, pose confidently against a vibrant graffiti-covered wall. Under the bright midday sun, they interact playfully with one another, some glancing around curiously while others nuzzle affectionately. The scene's dynamic energy is heightened by the stark interplay of light and shadow, creating an atmosphere that blends urban edginess with whimsical charm.", 
    "一只充满活力的动画兔子，身穿俏皮的粉色滑雪服，在湛蓝的天空下，熟练地从积雪的山坡上滑下。兔子充满活力地跳跃和旋转，在闪闪发光的雪地上投下动态阴影，而阳光的明亮光线则凸显了闪闪发光的景观，营造出一种欢快的氛围。当兔子下降时，它的流畅动作被广角镜头捕捉到，增加了速度感和刺激感。", 
    "A vibrant animated rabbit, dressed in a playful pink snowboarding outfit, expertly carves its way down a snowy mountain slope under a clear blue sky. The rabbit performs energetic jumps and spins, casting dynamic shadows on the glistening snow, while the sun's bright rays highlight the sparkling landscape, creating an atmosphere of joyful exhilaration. As the rabbit descends, its fluid motions are captured in a sweeping camera angle, adding to the sense of speed and excitement.", 
    "食物镜头，完美的汉堡，配上奶酪和生菜，微距拍摄，旋转拍摄，推拉镜头", 
    "food shot, a perfect burger in a bun with cheese and lettuce, macro shot, rotating shot, dolly in",  
    "这幅肖像画描绘了一只长着蓝眼睛的橘色猫，缓缓旋转，灵感来自维米尔的《戴珍珠耳环的少女》。这只猫戴着珍珠耳环，棕色的皮毛像荷兰帽一样，背景为黑色，在工作室灯光的映衬下显得格外明亮。", 
    "This portrait depicts an orange cat with blue eyes, slowly rotating, inspired by Vermeer ’s ’Girl with a Pearl Earring’. The cat is adorned with pearl earrings and has brown fur styled like a Dutch cap against a black background, illuminated by studio lighting.", 
    "一只熊猫在竹林下弹奏吉他，它的爪子轻轻拨动琴弦，一群着迷的兔子观看着，音乐与竹叶的沙沙声融为一体。高清。",  
    "A panda strumming a guitar under a bamboo grove, its paws gently plucking the strings as a group of mesmerized rabbits watch, the music blending with the rustle of bamboo leaves. HD.", 
    "雪花玻璃球摇晃后，会呈现出一座微型城市，雪花实际上是闪闪发光的星星。建筑物亮起，反射着天上的雪花，微小的人影在街道上移动，他们的路径被柔和的星光照亮，营造出神奇、宁静的城市景观。高清。", 
    "A snow globe, when shaken, reveals a miniature city where the snowflakes are actually glowing stars. The buildings light up, reflecting the celestial snowfall, and tiny figures move through the streets, their paths illuminated by the gentle starlight, creating a magical, peaceful urban landscape. HD.",  
    "魔术师水晶球的特写，展现了水晶球内部的未来城市景观。摩天大楼的光影直冲云霄，飞行汽车在空中飞驰，在水晶球表面投射出霓虹灯的反光。8K。", 
    "A close-up of a magician’s crystal ball that reveals a futuristic cityscape within. Skyscrapers of light stretch towards the heavens, and flying cars zip through the air, casting neon reflections across the ball’s surface. 8K.", 
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
